"""
CJ 대한통운 미래기술 챌린지
경로 최적화 + 적재 최적화 통합 솔루션
"""

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
    """CJ 대한통운 경로 및 적재 최적화 클래스"""
    
    def __init__(self):
        # 트럭 규격 (width x height x depth)
        self.truck_dimensions = (160, 280, 180)
        self.truck_capacity = 160 * 280 * 180
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        
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
        self.load_factor = 0.7
        num_vehicles = math.ceil(total_volume / (self.truck_capacity * self.load_factor))

        
        print(f"📊 총 부피: {total_volume:,}")
        print(f"🚛 필요 차량 수: {num_vehicles}")
        
        # PyVRP 모델 생성
        m = Model()
        m.add_vehicle_type(
            num_available=num_vehicles + 2, 
            capacity=int(self.truck_capacity * 0.7), 
            fixed_cost=150000, 
            unit_distance_cost=500
        )
        
        # 인덱스 매핑 생성
        self.index_to_order_number = []
        self.index_to_location_name = []
        
        # 창고(Depot) 추가
        depot = m.add_depot(x=coords[0][0], y=coords[0][1], name="Depot")
        self.index_to_order_number.append(self.df['Order_Number'][0])
        self.index_to_location_name.append("Depot")
        
        # 배송지 추가
        for idx, row in enumerate(self.df.iter_rows(named=True)):
            if idx != 0:  # 첫 번째는 창고이므로 제외
                m.add_client(
                    x=coords[idx][0],
                    y=coords[idx][1],
                    delivery=row['Volume'],
                    name=row['Destination']
                )
                self.index_to_order_number.append(row['Order_Number'])
                self.index_to_location_name.append(row['Destination'])
        
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
                    distance = distance_dict.get((origin, destination))
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
        """경로 결과 처리"""
        new_df = self.df.clear()
        
        for vehicle_id, route in enumerate(routes):
            print(f"▶ Vehicle {vehicle_id} route: {route}")
            if not route:
                continue
            
            route_str = [str(x) for x in route]
            order_map = {s: idx for idx, s in enumerate(route_str)}

            print(f"▶ order_map: {order_map}")
            
            # Depot 제외
            filtered_df = (
                self.df
                .filter(
                    pl.col('Order_Number').is_in(route_str)
                )
                .with_columns(
                    pl.col('Order_Number').map_elements(
                        lambda x: order_map.get(x, float('inf')),
                        return_dtype=pl.Int64
                    ).alias('route_order')
                )
                .sort('route_order')
                .drop('route_order')
            )
            
            # 창고 행 생성
            depot_row = pl.DataFrame({
                col: [None] if col != 'Destination' else ['Depot'] 
                for col in filtered_df.columns
            }).cast(dict(zip(filtered_df.columns, filtered_df.dtypes)))
            
            # 창고 -> 배송지들 -> 창고 순서로 결합
            result = pl.concat([depot_row, filtered_df, depot_row])
            
            # Route_Order와 Vehicle_ID 추가
            result = result.with_columns([
                (pl.int_range(pl.len()) + 1).alias('Route_Order'),
                pl.lit(vehicle_id).alias('Vehicle_ID').cast(pl.Int64)
            ])
            
            new_df = pl.concat([new_df, result])
        
        self.route_df = new_df
        
    def load_optimization(self):
        """적재 최적화 수행"""
        print("\n📦 적재 최적화 시작...")
        
        all_results = []
        
        # 각 차량별로 적재 최적화 수행
        vehicle_ids = self.route_df.select('Vehicle_ID').unique().to_series().to_list()
        
        for vehicle_id in vehicle_ids:
            if vehicle_id is None:
                continue
                
            print(f"🚛 Vehicle {vehicle_id} 적재 최적화...")
            
            # 해당 차량의 배송지만 필터링 (창고 제외)
            vehicle_items = self.route_df.filter(
                (pl.col('Vehicle_ID') == vehicle_id) & 
                (pl.col('Destination') != 'Depot')
            )
            
            if len(vehicle_items) == 0:
                continue
            
            # Stacking_Order는 route_order 역순으로 설정
            vehicle_items = vehicle_items.with_columns(
                pl.int_range(vehicle_items.height, 0, -1).alias('Stacking_Order')
            )
            
            # 3D 빈 패킹 수행
            packer = Packer()
            
            # 트럭 빈 생성 (width, height, depth, max_weight, max_items)
            # 트럭 빈 생성
            truck = Bin(
                partno='Truck',
                WHD=(self.truck_dimensions[0], self.truck_dimensions[1], self.truck_dimensions[2]),
                max_weight=999999,
                put_type=1
            )
            packer.addBin(truck)

            # 아이템 추가
            for row in vehicle_items.iter_rows(named=True):
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
                    level=row["Stacking_Order"],
                    updown=True,
                    loadbear=999999,
                    color='#FFCC00'
                )
                packer.addItem(item)

            # 패킹 실행
            packer.pack(
                fix_point=True,
                check_stable=False,
                bigger_first=False
            )
            
            # 결과 수집
            for item in packer.bins[0].items:
                all_results.append({
                    "Vehicle_ID": vehicle_id,
                    "Box_ID": item.name,
                    "Lower_Left_X": item.position[0],
                    "Lower_Left_Y": item.position[2],  # Y와 Z 좌표 교환
                    "Lower_Left_Z": item.position[1],
                    "Box_Width": item.width,
                    "Box_Length": item.depth,
                    "Box_Height": item.height
                })
        
        # 적재 결과 DataFrame 생성
        self.load_df = pl.DataFrame(all_results).with_columns([
            pl.col("Lower_Left_X").cast(pl.Float64),
            pl.col("Lower_Left_Y").cast(pl.Float64),
            pl.col("Lower_Left_Z").cast(pl.Float64),
            pl.col("Box_Width").cast(pl.Float64),
            pl.col("Box_Length").cast(pl.Float64),
            pl.col("Box_Height").cast(pl.Float64),
        ])
        
        print(f"✅ 적재 최적화 완료 - {len(self.load_df)}개 박스 배치")
        
    def save_results(self, output_file='Result.xlsx'):
        """결과를 Excel 파일로 저장"""
        print(f"\n💾 결과 저장 중: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 경로 최적화 결과
            route_pandas = self.route_df.to_pandas()
            route_pandas.to_excel(writer, sheet_name='Route_Optimization', index=False)
            
            # 적재 최적화 결과
            load_pandas = self.load_df.to_pandas()
            load_pandas.to_excel(writer, sheet_name='Load_Optimization', index=False)
            
            # 요약 정보
            summary_data = {
                'Metric': [
                    'Total Orders', 
                    'Total Vehicles', 
                    'Total Volume', 
                    'Average Load per Vehicle',
                    'Optimization Method'
                ],
                'Value': [
                    len(self.df) - 1,  # 창고 제외
                    len(self.route_df.select('Vehicle_ID').unique()) - 1,  # None 제외
                    self.df.select('Volume').sum()[0, 0],
                    self.df.select('Volume').sum()[0, 0] / (len(self.route_df.select('Vehicle_ID').unique()) - 1),
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
            
            # 2. 경로 최적화
            self.route_optimization()
            
            # 3. 적재 최적화
            self.load_optimization()
            
            # 4. 결과 저장
            self.save_results(output_file)
            
            print("\n" + "=" * 50)
            print("🎉 최적화 완료!")
            print(f"📊 총 {len(self.route_df.select('Vehicle_ID').unique()) - 1}대 차량")
            print(f"📦 총 {len(self.load_df)}개 박스 최적 배치")
            print(f"📁 결과 파일: {output_file}")
            print("=== self.route_df ===")
            print(self.route_df)

            print("=== self.load_df ===")
            print(self.load_df)
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    optimizer = CJOptimizer()
    optimizer.run_optimization(
        data_file=r"cj_challenge/main/data.json",
        distance_file=r"cj_challenge/main/distance-data.txt",
        output_file='Result.xlsx'
    )


if __name__ == "__main__":
    main()
