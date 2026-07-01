import pandas as pd
from typing import Tuple

def clean_data(df_desires: pd.DataFrame, df_metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Thực hiện làm sạch dữ liệu: xóa cột không dùng và xử lý thiếu dữ liệu.
    """
    # 1. Xóa cột không dùng
    cols_to_drop_desires = ['Other Reason for Automation Desire', 'Other Reason for Human Agency']
    df_desires = df_desires.drop(columns=cols_to_drop_desires, errors='ignore')
    
    if 'Zip Code' in df_metadata.columns:
        df_metadata = df_metadata.drop(columns=['Zip Code'])
        print('...Đã xóa cột Zip Code khỏi df_metadata')

    # 2. Impute các cột LLM Usage
    llm_cols = [c for c in df_metadata.columns if 'LLM Usage' in c]
    
    # Kiểm tra số liệu trước khi impute (tùy chọn: in ra màn hình hoặc log)
    num_missing_all = df_metadata[llm_cols].isna().all(axis=1).sum()
    print(f'Số dòng thiếu TẤT CẢ {len(llm_cols)} cột LLM Usage: {num_missing_all}')
    
    # Impute
    for col in llm_cols:
        df_metadata[col] = df_metadata[col].fillna('Không sử dụng')
    
    print('Đã impute xong các cột LLM Usage bằng "Không sử dụng"!')
    
    return df_desires, df_metadata

import pandas as pd
from typing import List, Tuple

def filter_cs_occupations(df_task, df_desires, df_metadata, df_expert):
    # 1. Lọc bảng task (giữ nguyên là DataFrame)
    cs_task = df_task[df_task['O*NET-SOC Code'].str.startswith('15-', na=False)]
    
    # 2. Lấy danh sách tên nghề (chỉ dùng để lọc các bảng khác)
    cs_occupations_list = cs_task['Occupation (O*NET-SOC Title)'].unique().tolist()
    
    # 3. Lọc các bảng khác (giữ nguyên là DataFrame)
    cs_desires = df_desires[df_desires['Occupation (O*NET-SOC Title)'].isin(cs_occupations_list)]
    cs_metadata = df_metadata[df_metadata['Occupation (O*NET-SOC Title)'].isin(cs_occupations_list)]
    cs_expert = df_expert[df_expert['Occupation (O*NET-SOC Title)'].isin(cs_occupations_list)]
    
    # Chuyển list occupations thành DataFrame để lưu được
    cs_occupations_df = pd.DataFrame(cs_occupations_list, columns=['Occupation'])
    
    return cs_desires, cs_metadata, cs_expert, cs_task, cs_occupations_df
