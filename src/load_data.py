import pandas as pd
from pathlib import Path
from typing import Optional

def load_data(file_name: str, file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Tải tệp dữ liệu từ thư mục data/raw/ của dự án.
    """
    if file_path is None:
        # Cách tìm project_root ổn định nhất: 
        # Tệp này nằm ở src/data/, vậy cần lùi 2 cấp để về gốc dự án
        project_root = Path(__file__).resolve().parent.parent
        file_path = project_root / "data" / "raw" / file_name
    else:
        file_path = Path(file_path)
    
    if not file_path.exists():
        # Thêm thông báo chi tiết để dễ debug
        raise FileNotFoundError(
            f"Không tìm thấy tệp tại: {file_path}. "
            f"Đảm bảo đường dẫn project_root là: {project_root}"
        )
        
    return pd.read_csv(file_path)

def get_data_info(df: pd.DataFrame, columns_to_check: Optional[list] = None) -> dict:
    """Trả về thông tin cơ bản của DataFrame."""
    info = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum(),
    }
    # Tự động lấy giá trị unique nếu người dùng truyền danh sách cột
    if columns_to_check:
        info["unique_values"] = {col: df[col].unique().tolist() for col in columns_to_check if col in df.columns}
        
    return info

def save_data(df: pd.DataFrame, file_name: str, folder_name: str = "processed") -> str:
    """
    Lưu DataFrame thành file csv vào thư mục data/processed/
    """
    # Xác định đường dẫn: project_root / data / processed / file_name
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / 'processed'
    
    # Tạo thư mục nếu nó chưa tồn tại
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / file_name
    df.to_csv(file_path, index=False)
    
    print(f"...Đã lưu tệp tại: {file_path}")
    return str(file_path)