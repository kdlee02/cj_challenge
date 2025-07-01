"""
CJ 대한통운 미래기술 챌린지
경로 최적화 + 적재 최적화 통합 솔루션
"""

import os
import json
import pandas as pd
import polars as pl
import numpy as np
import math
import openpyxl
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from pyproj import Transformer
from py3dbp import Packer, Bin, Item
import warnings
warnings.filterwarnings('ignore')


class CJOptimizer:
    """경로 및 적재 최적화 통합 클래스"""
    
    def __init__(self):
        # 트럭 규격 (width x height x depth) - Right-handed coordinate system
        self.truck_dimensions = (160, 280, 180)  # X, Y, Z
        self.truck_capacity = 160 * 280 * 180
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        self.load_factor = 0.7
        
    def load_data(self, data_file, distance_file):
        """데이터 파일 로드"""
        print("📂 데이터 로딩 중...")

        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        depot = json_data["depot"]
        destinations = json_data["destinations"]
        orders = json_data["orders"]

        # 1️⃣ Depot Row
        depot_row = {
            "Vehicle_ID": 0,
            "Route_Order": 0,
            "Destination": depot["destination"],
            "Order_Number": "DEPOT",
            "Box_ID": None,
            "Stacking_Order": None,
            "Lower_Left_X": 0,
            "Lower_Left_Y": 0,
            "Lower_Left_Z": 0,
            "Longitude": depot["location"]["longitude"],
            "Latitude": depot["location"]["latitude"],
            "Box_Width": 0,
            "Box_Length": 0,
            "Box_Height": 0,
            "Volume": 0
        }
        rows = [depot_row]

        # 2️⃣ destination_id → 좌표 매핑
        dest_coords = {
            d["destination_id"]: (d["location"]["longitude"], d["location"]["latitude"])
            for d in destinations
        }

        # 3️⃣ Orders Rows
        for order in orders:
            dest_id = order["destination"]
            longitude, latitude = dest_coords.get(dest_id, (None, None))
            width = order["dimension"]["width"]
            length = order["dimension"]["length"]
            height = order["dimension"]["height"]

            row = {
                "Vehicle_ID": 0,
                "Route_Order": 0,
                "Destination": dest_id,
                "Order_Number": order["order_number"],
                "Box_ID": order["box_id"],
                "Stacking_Order": 0,
                "Lower_Left_X": 0,
                "Lower_Left_Y": 0,
                "Lower_Left_Z": 0,
                "Longitude": longitude,
                "Latitude": latitude,
                "Box_Width": width,
                "Box_Length": length,
                "Box_Height": height,
                "Volume": width * length * height
            }
            rows.append(row)

        # 4️⃣ DataFrame 생성
        self.df = pl.DataFrame(rows)

        # 5️⃣ 거리 매트릭스
        self.matrix = pl.read_csv(distance_file, separator='\t')

        print(f"✅ 주문 데이터: {len(self.df)}건")
        print(f"✅ 거리 매트릭스: {len(self.matrix)}건")

    def route_optimization(self):
        """경로 최적화 수행"""
        print("\n🚛 경로 최적화 시작...")
        
        # 좌표 변환 (WGS84 -> Web Mercator)
        coords = [
            self.transformer.transform(lon, lat) 
            for lon, lat in zip(self.df['Longitude'], self.df['Latitude'])
        ]
        
        # 필요 차량 수 계산
        total_volume = self.df.select('Volume').sum()[0, 0]
        num_vehicles = math.ceil(total_volume / (self.truck_capacity * self.load_factor))
        
        print(f"📊 총 부피: {total_volume:,}")
        print(f"🚛 필요 차량 수: {num_vehicles}")
        
        # PyVRP 모델 생성
        m = Model()
        m.add_vehicle_type(
            num_available=num_vehicles + 2, 
            capacity=int(self.truck_capacity * self.load_factor), 
            fixed_cost=150000, 
            unit_distance_cost=500
        )
        
        # 인덱스와 실제 destination 매핑 생성
        self.index_to_destination = {}
        self.destination_to_index = {}
        
        # 창고(Depot) 추가
        depot = m.add_depot(x=coords[0][0], y=coords[0][1], name="Depot")
        self.index_to_destination[0] = "Depot"
        self.destination_to_index["Depot"] = 0
        
        # 배송지 추가
        unique_destinations = self.df.filter(pl.col('Destination') != 'Depot').select('Destination').unique()
        for idx, dest_row in enumerate(unique_destinations.iter_rows(named=True), 1):
            destination = dest_row['Destination']
            
            # 해당 destination의 좌표 찾기
            dest_data = self.df.filter(pl.col('Destination') == destination).row(0, named=True)
            dest_coords_idx = self.df.filter(pl.col('Destination') == destination).to_pandas().index[0]
            
            m.add_client(
                x=coords[dest_coords_idx][0],
                y=coords[dest_coords_idx][1],
                delivery=self.df.filter(pl.col('Destination') == destination).select('Volume').sum()[0, 0],
                name=destination
            )
            
            self.index_to_destination[idx] = destination
            self.destination_to_index[destination] = idx
        
        print(f"📍 인덱스 매핑: {self.index_to_destination}")
        
        # 거리 매트릭스 생성
        distance_dict = {}
        for row in self.matrix.iter_rows(named=True):
            key = (row['ORIGIN'], row['DESTINATION'])
            distance_dict[key] = row['DISTANCE_METER'] / 1000

        # 거리 매트릭스 추가
        for frm in m.locations:
            for to in m.locations:
                origin = frm.name
                destination = to.name
                if origin != destination:
                    distance = distance_dict.get((origin, destination), 999999)
                    m.add_edge(frm, to, distance=distance)
                else:
                    m.add_edge(frm, to, distance=0)
        
        # 경로 최적화 실행
        print("🔄 경로 최적화 실행 중... (최대 60초)")
        res = m.solve(stop=MaxRuntime(60), display=False)
        
        # 결과 처리
        routes = [list(route) for route in res.best.routes()]
        self.process_routes(routes)
        
        print(f"✅ 경로 최적화 완료 - {len(routes)}개 경로 생성")
        
    def process_routes(self, routes):
        """경로 결과 처리 및 적재 최적화"""
        print("\n📦 통합 최적화 처리 중...")
        
        final_results = []
        
        for vehicle_id, route in enumerate(routes):
            if not route:
                continue
                
            print(f"🚛 Vehicle {vehicle_id} 처리 중...")
            
            # 인덱스를 실제 destination으로 변환
            route_destinations = []
            for idx in route:
                destination = self.index_to_destination.get(idx, f"UNKNOWN_{idx}")
                route_destinations.append(destination)
            
            print(f"  Route destinations: {route_destinations}")
            
            # 해당 차량의 배송지 데이터 필터링
            vehicle_orders = []
            for route_order, destination in enumerate(route_destinations, 1):
                print(f"  처리 중인 destination: {destination}")
                
                # 해당 목적지의 모든 박스 찾기
                destination_boxes = self.df.filter(
                    pl.col('Destination') == destination
                ).to_dicts()
                
                print(f"  찾은 박스 수: {len(destination_boxes)}")
                
                for box in destination_boxes:
                    box_data = box.copy()
                    box_data['Vehicle_ID'] = vehicle_id
                    box_data['Route_Order'] = route_order
                    vehicle_orders.append(box_data)
            
            if not vehicle_orders:
                print(f"  ⚠️ Vehicle {vehicle_id}에 할당된 박스가 없습니다.")
                # 빈 차량이어도 창고 출발/도착은 추가
                depot_start = {
                    'Vehicle_ID': vehicle_id,
                    'Route_Order': 0,
                    'Destination': 'Depot',
                    'Order_Number': 'DEPOT',
                    'Box_ID': None,
                    'Stacking_Order': None,
                    'Lower_Left_X': 0,
                    'Lower_Left_Y': 0,
                    'Lower_Left_Z': 0,
                    'Longitude': self.df.filter(pl.col('Order_Number') == 'DEPOT')['Longitude'][0],
                    'Latitude': self.df.filter(pl.col('Order_Number') == 'DEPOT')['Latitude'][0],
                    'Box_Width': 0,
                    'Box_Length': 0,
                    'Box_Height': 0
                }
                
                depot_end = depot_start.copy()
                depot_end['Route_Order'] = len(route_destinations) + 1
                
                final_results.extend([depot_start, depot_end])
                continue
            
            # 적재 최적화 수행
            load_results = self.load_optimization_for_vehicle(vehicle_orders)
            
            # 창고 출발/도착 추가
            depot_start = {
                'Vehicle_ID': vehicle_id,
                'Route_Order': 0,
                'Destination': 'Depot',
                'Order_Number': 'DEPOT',
                'Box_ID': None,
                'Stacking_Order': None,
                'Lower_Left_X': 0,
                'Lower_Left_Y': 0,
                'Lower_Left_Z': 0,
                'Longitude': self.df.filter(pl.col('Order_Number') == 'DEPOT')['Longitude'][0],
                'Latitude': self.df.filter(pl.col('Order_Number') == 'DEPOT')['Latitude'][0],
                'Box_Width': 0,
                'Box_Length': 0,
                'Box_Height': 0
            }
            
            depot_end = depot_start.copy()
            depot_end['Route_Order'] = len(route_destinations) + 1
            
            # 결과 통합
            final_results.extend([depot_start])
            final_results.extend(load_results)
            final_results.extend([depot_end])
        
        # 빈 결과인 경우 기본 구조 생성
        if not final_results:
            print("⚠️ 최적화 결과가 없습니다. 기본 구조를 생성합니다.")
            final_results = [{
                'Vehicle_ID': 0,
                'Route_Order': 0,
                'Destination': 'Depot',
                'Order_Number': 'DEPOT',
                'Box_ID': None,
                'Stacking_Order': None,
                'Lower_Left_X': 0,
                'Lower_Left_Y': 0,
                'Lower_Left_Z': 0,
                'Longitude': 0,
                'Latitude': 0,
                'Box_Width': 0,
                'Box_Length': 0,
                'Box_Height': 0
            }]
        
        self.final_df = pl.DataFrame(final_results)
        print(f"✅ 최종 결과: {len(final_results)}개 레코드 생성")
        
    def load_optimization_for_vehicle(self, vehicle_orders):
        """개별 차량에 대한 적재 최적화"""
        
        # Route_Order 기준으로 정렬
        vehicle_orders.sort(key=lambda x: x['Route_Order'])
        
        # Stacking_Order 설정 (route_order 역순)
        max_route_order = max(order['Route_Order'] for order in vehicle_orders)
        for order in vehicle_orders:
            order['Stacking_Order'] = max_route_order - order['Route_Order'] + 1
        
        # 표준 py3dbp 사용
        packer = Packer()
        
        # 트럭 빈 생성 - name 파라미터 제거
        truck_bin = Bin(
            self.truck_dimensions[0],  # width
            self.truck_dimensions[1],  # height
            self.truck_dimensions[2],  # depth
            999999                     # max_weight
        )
        packer.add_bin(truck_bin)
        
        # 아이템 추가 - Stacking_Order 순으로 정렬 (낮은 번호가 먼저 적재)
        vehicle_orders.sort(key=lambda x: x['Stacking_Order'])
        
        for order in vehicle_orders:
            if order['Box_ID'] is None:
                continue
                
            item = Item(
                name=str(order['Box_ID']),
                width=int(order['Box_Width']),   # X축
                height=int(order['Box_Height']),  # Y축
                depth=int(order['Box_Length']),   # Z축
                weight=1
            )
            packer.add_item(item)
        
        # 패킹 수행
        packer.pack()
        
        # 결과 처리
        results = []
        packed_items = {}
        
        # 패킹된 아이템 정보 저장
        for bin_packed in packer.bins:
            for item in bin_packed.items:
                packed_items[item.name] = {
                    'Lower_Left_X': float(item.position[0]),
                    'Lower_Left_Y': float(item.position[1]), 
                    'Lower_Left_Z': float(item.position[2]),
                    'Box_Width': float(item.width),
                    'Box_Length': float(item.depth),
                    'Box_Height': float(item.height)
                }
        
        # 원래 순서로 결과 생성
        for order in vehicle_orders:
            if order['Box_ID'] is None:
                continue
                
            result = {
                'Vehicle_ID': order['Vehicle_ID'],
                'Route_Order': order['Route_Order'],
                'Destination': order['Destination'],
                'Order_Number': order['Order_Number'],
                'Box_ID': order['Box_ID'],
                'Stacking_Order': order['Stacking_Order'],
                'Longitude': order['Longitude'],
                'Latitude': order['Latitude']
            }
            
            # 패킹 결과 추가
            box_id_str = str(order['Box_ID'])
            if box_id_str in packed_items:
                result.update(packed_items[box_id_str])
            else:
                # 패킹되지 않은 경우 기본값
                result.update({
                    'Lower_Left_X': 0,
                    'Lower_Left_Y': 0,
                    'Lower_Left_Z': 0,
                    'Box_Width': float(order['Box_Width']),
                    'Box_Length': float(order['Box_Length']),
                    'Box_Height': float(order['Box_Height'])
                })
            
            results.append(result)
        
        return results
        
    def save_results(self, output_file='Result.xlsx'):
        """결과를 Excel 파일로 저장"""
        print(f"\n💾 결과 저장 중: {output_file}")
        
        # 컬럼 순서 정의
        column_order = [
            'Vehicle_ID', 'Route_Order', 'Destination', 'Order_Number', 'Box_ID',
            'Stacking_Order', 'Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z',
            'Longitude', 'Latitude', 'Box_Width', 'Box_Length', 'Box_Height'
        ]
        
        # 컬럼 순서에 맞게 재정렬
        final_df_ordered = self.final_df.select(column_order)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 통합 결과를 하나의 시트에 저장
            final_pandas = final_df_ordered.to_pandas()
            final_pandas.to_excel(writer, sheet_name='Detailed Route Information', index=False)
            
            # 요약 정보
            total_vehicles = len(self.final_df.filter(pl.col('Vehicle_ID').is_not_null()).select('Vehicle_ID').unique()) - 1
            total_orders = len(self.final_df.filter(pl.col('Box_ID').is_not_null()))
            
            summary_data = {
                'Metric': [
                    'Total Orders', 
                    'Total Vehicles', 
                    'Total Volume (cm³)', 
                    'Average Load per Vehicle (cm³)',
                    'Load Factor',
                    'Optimization Method'
                ],
                'Value': [
                    total_orders,
                    total_vehicles,
                    int(self.df.filter(pl.col('Order_Number') != 'DEPOT').select('Volume').sum()[0, 0]),
                    int(self.df.filter(pl.col('Order_Number') != 'DEPOT').select('Volume').sum()[0, 0] / max(total_vehicles, 1)),
                    f"{self.load_factor:.1%}",
                    'PyVRP + Py3DBP'
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"✅ 결과 저장 완료: {output_file}")
        
    def run_optimization(self, data_file='data.json', distance_file='distance-data.txt', output_file='Result.xlsx'):
        """전체 최적화 프로세스 실행"""
        print("🚀 CJ 대한통운 경로 및 적재 최적화 시작")
        print("=" * 50)
        
        try:
            # 1. 데이터 로드
            self.load_data(data_file, distance_file)
            
            # 2. 경로 최적화 및 적재 최적화 통합 수행
            self.route_optimization()
            
            # 3. 결과 저장
            self.save_results(output_file)
            
            print("\n" + "=" * 50)
            print("🎉 최적화 완료!")
            
            # 결과 통계
            total_vehicles = len(self.final_df.filter(pl.col('Vehicle_ID').is_not_null()).select('Vehicle_ID').unique())
            total_boxes = len(self.final_df.filter(pl.col('Box_ID').is_not_null()))
            
            print(f"📊 총 {total_vehicles}대 차량 사용")
            print(f"📦 총 {total_boxes}개 박스 최적 배치")
            print(f"📁 결과 파일: {output_file}")
            
            # 샘플 결과 출력
            print("\n📋 결과 샘플:")
            sample_df = self.final_df.head(10)
            print(sample_df.to_pandas().to_string(index=False))
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """메인 실행 함수"""
    optimizer = CJOptimizer()
    
    # 현재 스크립트가 있는 디렉토리 기준으로 파일 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data.json")
    distance_file = os.path.join(current_dir, "distance-data.txt")
    
    print(f"🔍 현재 디렉토리: {current_dir}")
    print(f"📄 데이터 파일: {data_file}")
    print(f"📄 거리 파일: {distance_file}")
    
    # 파일 존재 여부 확인
    if not os.path.exists(data_file):
        print(f"❌ 파일을 찾을 수 없습니다: {data_file}")
        return
    if not os.path.exists(distance_file):
        print(f"❌ 파일을 찾을 수 없습니다: {distance_file}")
        return
    
    optimizer.run_optimization(
        data_file=data_file,
        distance_file=distance_file,
        output_file='Result.xlsx'
    )


if __name__ == "__main__":
    main()