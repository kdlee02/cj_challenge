import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any

class VehicleLoadingValidator:
    def __init__(self, max_width=160, max_length=280, max_height=180):
        """
        차량 적재 제약조건 검증기 초기화
        
        Args:
            max_width: 차량 최대 너비 (X축, cm)
            max_length: 차량 최대 길이 (Y축, cm) 
            max_height: 차량 최대 높이 (Z축, cm)
        """
        self.MAX_WIDTH = max_width
        self.MAX_LENGTH = max_length
        self.MAX_HEIGHT = max_height
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Excel 파일에서 데이터 로드
        
        Args:
            filename: Excel 파일명
            
        Returns:
            박스 데이터가 포함된 DataFrame
        """
        df = pd.read_excel(filename)
        
        # Depot이 아닌 실제 박스 데이터만 필터링
        box_data = df[df['Box_ID'].notna() & df['Stacking_Order'].notna()].copy()
        
        return box_data
    
    def group_by_vehicle(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        차량별로 데이터 그룹화
        
        Args:
            df: 박스 데이터 DataFrame
            
        Returns:
            차량 ID를 키로 하는 딕셔너리
        """
        return {vehicle_id: group for vehicle_id, group in df.groupby('Vehicle_ID')}
    
    def boxes_overlap(self, box1: pd.Series, box2: pd.Series) -> bool:
        """
        두 박스가 3D 공간에서 겹치는지 확인
        
        Args:
            box1, box2: 박스 정보가 담긴 pandas Series
            
        Returns:
            겹치면 True, 겹치지 않으면 False
        """
        # Box1의 좌표 범위
        x1_min, x1_max = box1['Lower_Left_X'], box1['Lower_Left_X'] + box1['Box_Width']
        y1_min, y1_max = box1['Lower_Left_Y'], box1['Lower_Left_Y'] + box1['Box_Length']
        z1_min, z1_max = box1['Lower_Left_Z'], box1['Lower_Left_Z'] + box1['Box_Height']
        
        # Box2의 좌표 범위
        x2_min, x2_max = box2['Lower_Left_X'], box2['Lower_Left_X'] + box2['Box_Width']
        y2_min, y2_max = box2['Lower_Left_Y'], box2['Lower_Left_Y'] + box2['Box_Length']
        z2_min, z2_max = box2['Lower_Left_Z'], box2['Lower_Left_Z'] + box2['Box_Height']
        
        # 3D에서 겹치지 않는 조건: 어느 한 축에서라도 완전히 분리되어 있으면 겹치지 않음
        separated_x = x1_max <= x2_min or x2_max <= x1_min
        separated_y = y1_max <= y2_min or y2_max <= y1_min
        separated_z = z1_max <= z2_min or z2_max <= z1_min
        
        # 하나라도 분리되어 있으면 겹치지 않음
        return not (separated_x or separated_y or separated_z)
    
    def box_exceeds_bounds(self, box: pd.Series) -> Dict[str, Any]:
        """
        박스가 차량 경계를 벗어나는지 확인
        
        Args:
            box: 박스 정보가 담긴 pandas Series
            
        Returns:
            경계 초과 정보가 담긴 딕셔너리
        """
        right_edge = box['Lower_Left_X'] + box['Box_Width']
        back_edge = box['Lower_Left_Y'] + box['Box_Length']
        top_edge = box['Lower_Left_Z'] + box['Box_Height']
        
        return {
            'exceeds_width': right_edge > self.MAX_WIDTH,
            'exceeds_length': back_edge > self.MAX_LENGTH,
            'exceeds_height': top_edge > self.MAX_HEIGHT,
            'right_edge': right_edge,
            'back_edge': back_edge,
            'top_edge': top_edge
        }
    
    def validate_vehicle(self, vehicle_id: int, boxes_df: pd.DataFrame) -> Dict[str, Any]:
        """
        단일 차량의 제약조건 검증
        
        Args:
            vehicle_id: 차량 ID
            boxes_df: 해당 차량의 박스 데이터 DataFrame
            
        Returns:
            검증 결과가 담긴 딕셔너리
        """
        violations = {
            'vehicle_id': vehicle_id,
            'overlapping_boxes': [],
            'boundary_violations': [],
            'is_valid': True
        }
        
        boxes_list = boxes_df.reset_index(drop=True)
        
        # 1. 박스 간 겹침 검사
        for i in range(len(boxes_list)):
            for j in range(i + 1, len(boxes_list)):
                if self.boxes_overlap(boxes_list.iloc[i], boxes_list.iloc[j]):
                    box1, box2 = boxes_list.iloc[i], boxes_list.iloc[j]
                    violations['overlapping_boxes'].append({
                        'box1': {
                            'id': box1['Box_ID'],
                            'position': [box1['Lower_Left_X'], box1['Lower_Left_Y'], box1['Lower_Left_Z']],
                            'size': [box1['Box_Width'], box1['Box_Length'], box1['Box_Height']]
                        },
                        'box2': {
                            'id': box2['Box_ID'],
                            'position': [box2['Lower_Left_X'], box2['Lower_Left_Y'], box2['Lower_Left_Z']],
                            'size': [box2['Box_Width'], box2['Box_Length'], box2['Box_Height']]
                        }
                    })
                    violations['is_valid'] = False
        
        # 2. 차량 경계 초과 검사
        for _, box in boxes_list.iterrows():
            bound_check = self.box_exceeds_bounds(box)
            if any([bound_check['exceeds_width'], bound_check['exceeds_length'], bound_check['exceeds_height']]):
                violations['boundary_violations'].append({
                    'box_id': box['Box_ID'],
                    'position': [box['Lower_Left_X'], box['Lower_Left_Y'], box['Lower_Left_Z']],
                    'size': [box['Box_Width'], box['Box_Length'], box['Box_Height']],
                    'actual_bounds': [bound_check['right_edge'], bound_check['back_edge'], bound_check['top_edge']],
                    'max_bounds': [self.MAX_WIDTH, self.MAX_LENGTH, self.MAX_HEIGHT],
                    'violations': {
                        'width': bound_check['exceeds_width'],
                        'length': bound_check['exceeds_length'],
                        'height': bound_check['exceeds_height']
                    }
                })
                violations['is_valid'] = False
        
        return violations
    
    def validate_all(self, filename: str) -> Dict[str, Any]:
        """
        전체 데이터 검증
        
        Args:
            filename: Excel 파일명
            
        Returns:
            전체 검증 결과가 담긴 딕셔너리
        """
        data = self.load_data(filename)
        vehicles = self.group_by_vehicle(data)
        
        results = {
            'total_vehicles': len(vehicles),
            'valid_vehicles': 0,
            'invalid_vehicles': 0,
            'vehicle_results': []
        }
        
        for vehicle_id, boxes_df in vehicles.items():
            validation = self.validate_vehicle(vehicle_id, boxes_df)
            results['vehicle_results'].append(validation)
            
            if validation['is_valid']:
                results['valid_vehicles'] += 1
            else:
                results['invalid_vehicles'] += 1
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        결과를 보기 좋게 출력
        
        Args:
            results: validate_all()의 결과
        """
        print('=' * 60)
        print('차량 적재 제약조건 검증 결과')
        print('=' * 60)
        print(f'총 차량 수: {results["total_vehicles"]}')
        print(f'유효한 차량: {results["valid_vehicles"]}')
        print(f'제약조건 위반 차량: {results["invalid_vehicles"]}')
        print()
        
        for vehicle in results['vehicle_results']:
            if not vehicle['is_valid']:
                print(f'🚚 차량 ID: {vehicle["vehicle_id"]} - ❌ 제약조건 위반')
                
                if vehicle['overlapping_boxes']:
                    print(f'  📦 겹치는 박스 쌍: {len(vehicle["overlapping_boxes"])}개')
                    for idx, overlap in enumerate(vehicle['overlapping_boxes']):
                        print(f'    {idx + 1}. {overlap["box1"]["id"]} ↔ {overlap["box2"]["id"]}')
                        print(f'       {overlap["box1"]["id"]}: 위치{overlap["box1"]["position"]} 크기{overlap["box1"]["size"]}')
                        print(f'       {overlap["box2"]["id"]}: 위치{overlap["box2"]["position"]} 크기{overlap["box2"]["size"]}')
                
                if vehicle['boundary_violations']:
                    print(f'  🚫 경계 초과 박스: {len(vehicle["boundary_violations"])}개')
                    for idx, violation in enumerate(vehicle['boundary_violations']):
                        print(f'    {idx + 1}. {violation["box_id"]}')
                        print(f'       위치: {violation["position"]}, 크기: {violation["size"]}')
                        print(f'       실제 경계: {violation["actual_bounds"]}')
                        print(f'       최대 허용: {violation["max_bounds"]}')
                        violation_types = []
                        if violation['violations']['width']:
                            violation_types.append('너비 초과')
                        if violation['violations']['length']:
                            violation_types.append('길이 초과')
                        if violation['violations']['height']:
                            violation_types.append('높이 초과')
                        print(f'       위반 유형: {", ".join(violation_types)}')
                print()
            else:
                print(f'🚚 차량 ID: {vehicle["vehicle_id"]} - ✅ 제약조건 만족')

def main():
    """
    메인 실행 함수 - 사용 예시
    """
    # 검증기 생성 (최대 적재 부피: 160x280x180cm)
    validator = VehicleLoadingValidator(max_width=160, max_length=280, max_height=180)
    
    try:
        print('차량 적재 제약조건 검증을 시작합니다...')
        
        # 검증 실행
        results = validator.validate_all('C:/Users/Grace/Desktop/2025/프로젝트/미래기술챌린지/cj_challenge/Result.xlsx')
        
        # 결과 출력
        validator.print_results(results)
        
        # 추가 통계 정보
        print('\n📊 추가 통계:')
        total_overlaps = sum(len(v['overlapping_boxes']) for v in results['vehicle_results'])
        total_boundary_violations = sum(len(v['boundary_violations']) for v in results['vehicle_results'])
        
        print(f'총 박스 겹침 건수: {total_overlaps}')
        print(f'총 경계 초과 건수: {total_boundary_violations}')
        
        if results['invalid_vehicles'] == 0:
            print('\n🎉 모든 차량이 제약조건을 만족합니다!')
        else:
            print(f'\n⚠️  {results["invalid_vehicles"]}대의 차량에서 제약조건 위반이 발견되었습니다.')
        
        return results
        
    except Exception as e:
        print(f'검증 중 오류 발생: {e}')
        return None

if __name__ == '__main__':
    main()