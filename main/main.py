import json
import pandas as pd
import polars as pl
import numpy as np
import math
import os
from collections import defaultdict
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from pyproj import Transformer
from py3dbp import Packer, Bin, Item

def load_and_process_data():
    """데이터 로드 및 전처리"""
    import os
    
    # 현재 스크립트의 디렉토리 기준으로 상대경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, '..', 'dataset', 'Data_Set.json')
    txt_path = os.path.join(script_dir, '..', 'dataset', 'distance-data.txt')
    
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터프레임 생성
    rows = []
    
    # Depot 추가
    depot = data['depot']
    rows.append({
        'Vehicle_ID': 0,
        'Route_Order': 0,
        'Destination': 'Depot',
        'Order_Number': None,
        'Box_ID': None,
        'Stacking_Order': None,
        'Lower_Left_X': None,
        'Lower_Left_Y': None,
        'Lower_Left_Z': None,
        'Longitude': depot['location']['longitude'],
        'Latitude': depot['location']['latitude'],
        'Box_Width': 0,
        'Box_Length': 0,
        'Box_Height': 0,
        'Volume': 0
    })
    
    # 주문 정보 추가
    for order in data['orders']:
        # 해당 목적지 찾기
        dest_info = next(d for d in data['destinations'] if d['destination_id'] == order['destination'])
        
        volume = order['dimension']['width'] * order['dimension']['length'] * order['dimension']['height']
        
        rows.append({
            'Vehicle_ID': 0,
            'Route_Order': 0,
            'Destination': order['destination'],
            'Order_Number': order['order_number'],
            'Box_ID': order['box_id'],
            'Stacking_Order': 0,
            'Lower_Left_X': 0,
            'Lower_Left_Y': 0,
            'Lower_Left_Z': 0,
            'Longitude': dest_info['location']['longitude'],
            'Latitude': dest_info['location']['latitude'],
            'Box_Width': order['dimension']['width'],
            'Box_Length': order['dimension']['length'],
            'Box_Height': order['dimension']['height'],
            'Volume': volume
        })
    
    df = pl.DataFrame(rows)
    
    # 거리 데이터 로드
    matrix = pl.read_csv(txt_path, separator='\t')
    
    return df, matrix

def optimize_routes(df, matrix):
    """경로 최적화"""
    truck_capacity = 160 * 280 * 180
    
    # 좌표 변환
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    coords = [transformer.transform(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])]
    
    # 차량 수 계산
    num_vehicles = math.ceil(df.select('Volume').sum()[0, 0] / truck_capacity)
    
    # 모델 생성
    m = Model()
    m.add_vehicle_type(num_available=num_vehicles+2, capacity=int(truck_capacity*0.7), 
                      fixed_cost=150000, unit_distance_cost=500)
    
    # 인덱스 매핑
    index_to_order_number = []
    index_to_location_name = []
    
    # Depot 추가
    depot = m.add_depot(x=coords[0][0], y=coords[0][1], name="Depot")
    index_to_order_number.append(df['Order_Number'][0])
    index_to_location_name.append("Depot")
    
    # 고객 추가
    for idx, row in enumerate(df.iter_rows(named=True)):
        if idx != 0:
            xs = coords[idx][0]
            ys = coords[idx][1]
            deliver = row['Volume']
            m.add_client(
                x=xs,
                y=ys,
                delivery=deliver,
                name=row['Destination']
            )
            index_to_order_number.append(row['Order_Number'])
            index_to_location_name.append(row['Destination'])
    
    # 엣지 추가
    for frm in m.locations:
        for to in m.locations:
            origin = frm.name
            destination = to.name
            if origin != destination:
                distance_result = matrix.filter((pl.col('ORIGIN') == origin) & (pl.col('DESTINATION') == destination))
                if distance_result.height > 0:
                    distance = distance_result.select('DISTANCE_METER').to_series().item() / 1000
                else:
                    # 거리 정보가 없으면 유클리드 거리 사용
                    frm_coords = next(coord for i, coord in enumerate(coords) if index_to_location_name[i] == origin)
                    to_coords = next(coord for i, coord in enumerate(coords) if index_to_location_name[i] == destination)
                    distance = np.sqrt((frm_coords[0] - to_coords[0])**2 + (frm_coords[1] - to_coords[1])**2) / 1000
                m.add_edge(frm, to, distance=distance)
            else:
                m.add_edge(frm, to, distance=0)
    
    # 최적화 실행
    res = m.solve(stop=MaxRuntime(300), display=False)
    
    # 결과 처리
    routes = [list(route) for route in res.best.routes()]
    new_df = df.clear()
    
    for i in range(len(routes)):
        order_map = {order_num: idx for idx, order_num in enumerate(routes[i])}
        
        filtered_df = (df
            .filter(pl.col('Order_Number').is_in(routes[i]))
            .with_columns(
                pl.col('Order_Number').map_elements(
                    lambda x: order_map.get(x, float('inf')), 
                    return_dtype=pl.Int64
                ).alias('route_order')
            )
            .sort('route_order')
            .drop('route_order')
        )
        
        depot_row = pl.DataFrame({
            col: [None] if col != 'Destination' else ['Depot'] 
            for col in filtered_df.columns
        }).cast(dict(zip(filtered_df.columns, filtered_df.dtypes)))
        
        result = pl.concat([
            depot_row,           
            filtered_df,       
            depot_row          
        ])
        
        result = result.with_columns(
            (pl.int_range(pl.len()) + 1).alias('Route_Order')
        )
        
        result = result.with_columns(
            pl.lit(i).alias('Vehicle_ID').cast(pl.Int64)
        )
        new_df = pl.concat([new_df, result])
    
    return new_df

def create_bin(df, vehicle_number):
    """차량별 적재 최적화"""
    rows = df.filter(pl.col('Vehicle_ID') == vehicle_number).filter(pl.col('Destination') != 'Depot')
    rows = rows.with_columns(
        pl.int_range(rows.height, 0, -1).alias('rank')
    )
    
    packer = Packer()
    
    truck = Bin(
        partno='Truck',
        WHD=(160, 180, 280),
        max_weight=999999,
        put_type=1        
    )
    packer.addBin(truck)  
    
    for row in rows.iter_rows(named=True):
        item = Item(
            partno=row["Box_ID"],
            name=row["Box_ID"],
            typeof='cube',
            WHD=(
                int(row["Box_Width"]),
                int(row["Box_Height"]),
                int(row["Box_Length"])
            ),
            weight=1,
            level=row["rank"],      
            updown=True,
            loadbear=999999,
            color='#FFCC00'
        )
        packer.addItem(item)  
    
    packer.pack(
        fix_point=True,
        check_stable=False,
        bigger_first=False 
    )
    return truck, packer

def optimize_loading(df):
    """적재 최적화"""
    # Stacking_Order 계산 (vehicle별로 route_order 역순)
    df = df.with_columns(
        pl.col('Route_Order').max().over('Vehicle_ID').alias('max_route_order')
    ).with_columns(
        (pl.col('max_route_order') - pl.col('Route_Order') + 1).alias('Stacking_Order')
    ).drop('max_route_order')
    
    packing = defaultdict()
    unfitted = []
    
    max_vehicle_id = df.select(['Vehicle_ID']).max().item()
    
    for x in range(max_vehicle_id + 1):
        truck, packer = create_bin(df, x)
        unfitted.append(len(truck.unfitted_items))
        packing[x] = packer
    
    if sum(unfitted) == 0:
        print("All items fitted")
    else:
        print(f"Unfitted items: {sum(unfitted)}")
    
    # 적재 결과 데이터프레임 생성
    packing_df = pl.DataFrame()
    for i in range(max_vehicle_id + 1):
        records = []
        for item in packing[i].bins[0].items:
            records.append({
                "Box_ID": item.name,
                "Lower_Left_X": item.position[0],
                "Lower_Left_Y": item.position[2], 
                "Lower_Left_Z": item.position[1],
                "Box_Width": item.width,
                "Box_Length": item.depth,
                "Box_Height": item.height
            })
        
        if records:
            vehicle_packing = pl.DataFrame(records)
            vehicle_packing = vehicle_packing.with_columns([
                pl.col("Lower_Left_X").cast(pl.Float64),
                pl.col("Lower_Left_Y").cast(pl.Float64),
                pl.col("Lower_Left_Z").cast(pl.Float64),
                pl.col("Box_Width").cast(pl.Float64),
                pl.col("Box_Length").cast(pl.Float64),
                pl.col("Box_Height").cast(pl.Float64),
            ])
            packing_df = packing_df.vstack(vehicle_packing)
    
    # 원본 데이터와 적재 결과 조인
    joined = df.join(packing_df, on='Box_ID', how='left', suffix='_packing')
    
    # 좌표 컬럼 업데이트
    cols_to_replace = ['Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z']
    
    final_df = joined.with_columns([
        pl.coalesce([pl.col(f"{col}_packing"), pl.col(col)]).alias(col)
        for col in cols_to_replace
    ]).select([col for col in joined.columns if not col.endswith('_packing')])
    
    return final_df

def main():
    """메인 함수"""
    print("데이터 로드 중...")
    df, matrix = load_and_process_data()
    
    print("경로 최적화 중...")
    optimized_routes_df = optimize_routes(df, matrix)
    
    print("적재 최적화 중...")
    final_df = optimize_loading(optimized_routes_df)
    
    print("결과 저장 중...")
    final_df.write_excel('Result.xlsx')
    
    print("최적화 완료! Result.xlsx 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()