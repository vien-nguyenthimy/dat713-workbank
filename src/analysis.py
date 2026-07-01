import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_cs_desires(path="data/processed", desire_col="Automation Desire Rating",
                     agency_col="Human Agency Scale Rating"):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        print(f"Đã đọc file: {path} ({len(df)} dòng)")
        return df

    candidates = sorted(glob.glob(os.path.join(path, "cs_*.csv")))
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy file nào dạng 'cs_*.csv' trong thư mục '{path}'. "
            f"Hãy chỉ định rõ đường dẫn file, ví dụ: load_cs_desires('data/processed/cs_xxx.csv')"
        )

    for f in candidates:
        df = pd.read_csv(f)
        if desire_col in df.columns and agency_col in df.columns:
            print(f"Đã đọc file: {f} ({len(df)} dòng)")
            return df

    raise ValueError(
        f"Tìm thấy các file {candidates} nhưng không file nào có đủ 2 cột "
        f"'{desire_col}' và '{agency_col}'. Hãy chỉ định rõ đường dẫn file đúng."
    )


def check_paradox(df, desire_col="Automation Desire Rating", agency_col="Human Agency Scale Rating"):
    corr = df[desire_col].corr(df[agency_col])
    print(f"Tương quan Pearson (Desire vs Human Agency): r = {corr:.3f}")
    print(f"Số quan sát: {len(df)}")
    return corr


def classify_mindset_group(df, desire_col="Automation Desire Rating", agency_col="Human Agency Scale Rating"):
    df = df.copy()
    d_med = df[desire_col].median()
    a_med = df[agency_col].median()
    print(f"Ngưỡng trung vị: Desire = {d_med}, Agency = {a_med}")

    def _classify(row):
        d_high = row[desire_col] >= d_med
        a_high = row[agency_col] >= a_med
        if d_high and not a_high:
            return "Delegate hoan toan"
        elif d_high and a_high:
            return "Nghich ly uy quyen"
        elif not d_high and a_high:
            return "Giu chat"
        else:
            return "Tho o"

    df["Mindset_Group"] = df.apply(_classify, axis=1)

    counts = df["Mindset_Group"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print("\nPhân bố 4 nhóm tâm lý:")
    for g in counts.index:
        print(f"  {g}: {counts[g]} ({pct[g]}%)")

    return df


def plot_mindset_matrix(df, desire_col="Automation Desire Rating", agency_col="Human Agency Scale Rating",
                         group_col="Mindset_Group", save_path=None):
    if group_col not in df.columns:
        raise ValueError(f"Thiếu cột '{group_col}'. Hãy chạy classify_mindset_group(df) trước.")

    d_med = df[desire_col].median()
    a_med = df[agency_col].median()

    colors = {
        "Delegate hoan toan": "#4C9A2A",
        "Nghich ly uy quyen": "#E0A800",
        "Giu chat": "#C0392B",
        "Tho o": "#95A5A6",
    }

    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(7, 6))

    for g in df[group_col].value_counts().index:
        sub = df[df[group_col] == g]
        jitter_x = sub[desire_col] + np.random.uniform(-0.15, 0.15, size=len(sub))
        jitter_y = sub[agency_col] + np.random.uniform(-0.15, 0.15, size=len(sub))
        ax.scatter(jitter_x, jitter_y, alpha=0.15, s=40, color=colors.get(g, "gray"), label=g)

    ax.axvline(d_med, color="black", linestyle="--", linewidth=1)
    ax.axhline(a_med, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(desire_col)
    ax.set_ylabel(agency_col)
    ax.set_title("Ma trận tâm lý: Desire vs Human Agency")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    return fig

def _get_reason_cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def rank_reasons(df):
    """
    Xếp hạng tần suất (%) các lý do muốn tự động hóa và lý do muốn giữ quyền,
    tính trên toàn bộ df được truyền vào.

    Trả về tuple (auto_pct, agency_pct) — 2 pandas Series đã sort giảm dần.
    """
    auto_cols = _get_reason_cols(df, "Reasons for Automation Desire")
    agency_cols = _get_reason_cols(df, "Reasons for Human Agency")

    auto_pct = (df[auto_cols].sum() / len(df) * 100).sort_values(ascending=False)
    agency_pct = (df[agency_cols].sum() / len(df) * 100).sort_values(ascending=False)

    print("Lý do muốn TỰ ĐỘNG HÓA:")
    for r, v in auto_pct.items():
        print(f"  {r.replace('Reasons for Automation Desire - ', '')}: {v:.1f}%")

    print("\nLý do muốn GIỮ QUYỀN CON NGƯỜI:")
    for r, v in agency_pct.items():
        print(f"  {r.replace('Reasons for Human Agency - ', '')}: {v:.1f}%")

    return auto_pct, agency_pct


def paradox_group_reasons(df, group_col="Mindset_Group", group_name="Nghich ly uy quyen"):
    """
    Lọc riêng nhóm tâm lý chỉ định (mặc định: Nghịch lý ủy quyền) và xếp hạng
    lý do giữ quyền chiếm ưu thế trong nhóm đó.

    Trả về pandas Series (lý do -> %) đã sort giảm dần.
    """
    if group_col not in df.columns:
        raise ValueError(f"Thiếu cột '{group_col}'. Hãy chạy classify_mindset_group(df) trước (analysis_3_1.py).")

    sub = df[df[group_col] == group_name]
    agency_cols = _get_reason_cols(df, "Reasons for Human Agency")
    pct = (sub[agency_cols].sum() / len(sub) * 100).sort_values(ascending=False)

    print(f"Số quan sát nhóm '{group_name}': {len(sub)}")
    print(f"\nLý do giữ quyền chiếm ưu thế trong nhóm '{group_name}':")
    for r, v in pct.items():
        print(f"  {r.replace('Reasons for Human Agency - ', '')}: {v:.1f}%")

    return pct


def compare_groups_reasons(df, group_col="Mindset_Group",
                            group_a="Nghich ly uy quyen", group_b="Giu chat",
                            save_path=None):
    """
    So sánh trực quan (bar chart) tỷ lệ lý do giữ quyền giữa 2 nhóm tâm lý
    bất kỳ (mặc định: Nghịch lý ủy quyền vs Giữ chặt).

    Trả về DataFrame so sánh (đã dùng để vẽ).
    """
    if group_col not in df.columns:
        raise ValueError(f"Thiếu cột '{group_col}'. Hãy chạy classify_mindset_group(df) trước (analysis_3_1.py).")

    agency_cols = _get_reason_cols(df, "Reasons for Human Agency")

    sub_a = df[df[group_col] == group_a]
    sub_b = df[df[group_col] == group_b]

    pct_a = (sub_a[agency_cols].sum() / len(sub_a) * 100)
    pct_b = (sub_b[agency_cols].sum() / len(sub_b) * 100)

    compare_df = pd.DataFrame({group_a: pct_a, group_b: pct_b})
    compare_df.index = [i.replace("Reasons for Human Agency - ", "") for i in compare_df.index]
    compare_df = compare_df.sort_values(group_a, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    compare_df.plot(kind="barh", ax=ax, color=["#E0A800", "#C0392B"])
    ax.set_xlabel("% lựa chọn lý do")
    ax.set_title(f"So sánh động cơ giữ quyền: {group_a} vs {group_b}")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")

    plt.show()
    return compare_df
 
 
def occupation_mindset_table(df, occ_col="Occupation (O*NET-SOC Title)",
                              group_col="Mindset_Group", min_n=20):
    """
    Tính tỷ lệ (%) 4 nhóm tâm lý cho từng nghề trong ngành CS.
 
    Chỉ giữ lại các nghề có ít nhất `min_n` quan sát (mặc định 20) để tránh
    tỷ lệ % bị nhiễu do cỡ mẫu quá nhỏ.
 
    Trả về DataFrame: hàng = nghề, cột = 4 nhóm tâm lý, giá trị = %,
    đã sort giảm dần theo tỷ lệ 'Nghich ly uy quyen'.
    """
    if group_col not in df.columns:
        raise ValueError(f"Thiếu cột '{group_col}'. Hãy chạy classify_mindset_group(df) trước (analysis_3_1.py).")
 
    occ_counts = df[occ_col].value_counts()
    valid_occ = occ_counts[occ_counts >= min_n].index
 
    sub = df[df[occ_col].isin(valid_occ)]
 
    table = pd.crosstab(sub[occ_col], sub[group_col], normalize="index") * 100
    table = table.round(1)
    table["N"] = sub[occ_col].value_counts()
 
    if "Nghich ly uy quyen" in table.columns:
        table = table.sort_values("Nghich ly uy quyen", ascending=False)
 
    print(f"Số nghề đủ điều kiện (>= {min_n} quan sát): {len(table)}")
    print("\nBảng tỷ lệ 4 nhóm tâm lý theo nghề (%):")
    print(table)
 
    return table
 
 
def plot_occupation_heatmap(table, save_path=None):
    """
    Vẽ heatmap trực quan từ bảng tỷ lệ 4 nhóm tâm lý theo nghề
    (đầu ra của occupation_mindset_table, KHÔNG bao gồm cột 'N').
    """
    plot_cols = [c for c in table.columns if c != "N"]
    data = table[plot_cols]
 
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(data))))
    im = ax.imshow(data.values, cmap="YlOrRd", aspect="auto")
 
    ax.set_xticks(range(len(plot_cols)))
    ax.set_xticklabels(plot_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index)
 
    for i in range(len(data)):
        for j in range(len(plot_cols)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color="white" if val > data.values.max() * 0.6 else "black", fontsize=8)
 
    ax.set_title("Tỷ lệ (%) 4 nhóm tâm lý theo nghề trong ngành CS")
    fig.colorbar(im, ax=ax, label="% trong nghề")
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
 
    plt.show()
    return fig
 
 
def occupation_top_reason(df, occ_col="Occupation (O*NET-SOC Title)",
                           group_col="Mindset_Group", group_name="Nghich ly uy quyen",
                           min_n=10):
    """
    Với mỗi nghề, lọc riêng các dòng thuộc nhóm `group_name` (mặc định: Nghịch lý
    ủy quyền), rồi tìm lý do giữ quyền (Reasons for Human Agency - *) được chọn
    nhiều nhất trong nghề đó.
 
    Chỉ tính cho các nghề có ít nhất `min_n` quan sát trong nhóm đó.
 
    Trả về DataFrame: nghề -> lý do chiếm ưu thế, % tương ứng, cỡ mẫu n.
    """
    if group_col not in df.columns:
        raise ValueError(f"Thiếu cột '{group_col}'. Hãy chạy classify_mindset_group(df) trước (analysis_3_1.py).")
 
    agency_cols = [c for c in df.columns if c.startswith("Reasons for Human Agency")]
    sub = df[df[group_col] == group_name]
 
    rows = []
    for occ, g in sub.groupby(occ_col):
        if len(g) < min_n:
            continue
        pct = (g[agency_cols].sum() / len(g) * 100)
        top_reason = pct.idxmax().replace("Reasons for Human Agency - ", "")
        rows.append({
            "Occupation": occ,
            "Top_Reason": top_reason,
            "Top_Reason_Pct": round(pct.max(), 1),
            "N": len(g),
        })
 
    result = pd.DataFrame(rows).sort_values("Top_Reason_Pct", ascending=False).reset_index(drop=True)
 
    print(f"Lý do giữ quyền chiếm ưu thế theo nghề (chỉ trong nhóm '{group_name}', n >= {min_n}):")
    print(result.to_string(index=False))
 
    return result


def merge_desire_capacity(cs_desires, cs_expert, task_col="Task ID"):
    """
    Gộp dữ liệu Desire (worker) và Capacity (expert) ở cấp độ TASK (không phải
    cấp độ dòng đánh giá cá nhân) — tính trung bình các chỉ số quan trọng cho
    từng Task ID, vì mỗi task có thể được nhiều worker/expert đánh giá khác nhau.

    Trả về DataFrame ở cấp task, với các cột:
        Task ID, Occupation, Automation Desire Rating (mean), Human Agency Scale Rating (mean),
        Automation Capacity Rating (mean), Dominant_Agency_Reason (lý do giữ quyền phổ biến nhất
        trong các đánh giá thuộc task đó), n_desire, n_expert
    """
    agency_cols = [c for c in cs_desires.columns if c.startswith("Reasons for Human Agency")]

    desire_agg = cs_desires.groupby(task_col).agg(
        **{
            "Occupation": ("Occupation (O*NET-SOC Title)", "first"),
            "Automation Desire Rating": ("Automation Desire Rating", "mean"),
            "Human Agency Scale Rating": ("Human Agency Scale Rating", "mean"),
            "n_desire": ("Automation Desire Rating", "count"),
        }
    )

    # Lý do giữ quyền phổ biến nhất cho mỗi task (tổng số lần được chọn, lấy lý do nhiều nhất)
    reason_sum = cs_desires.groupby(task_col)[agency_cols].sum()
    dominant_reason = reason_sum.idxmax(axis=1).str.replace("Reasons for Human Agency - ", "", regex=False)
    desire_agg["Dominant_Agency_Reason"] = dominant_reason

    capacity_agg = cs_expert.groupby(task_col).agg(
        **{
            "Automation Capacity Rating": ("Automation Capacity Rating", "mean"),
            "n_expert": ("Automation Capacity Rating", "count"),
        }
    )

    task_df = desire_agg.join(capacity_agg, how="inner").reset_index()

    print(f"Số task có đủ cả dữ liệu Desire và Capacity: {len(task_df)}")
    return task_df


def classify_zone(df, desire_col="Automation Desire Rating", capacity_col="Automation Capacity Rating"):
    """
    Chia mỗi task vào 1 trong 4 vùng dựa trên ngưỡng trung vị của Desire và Capacity:

        - Vung an toan          : Desire cao, Capacity cao -> nên triển khai
        - Vung rui ro ky thuat  : Desire cao, Capacity thấp -> worker muốn nhưng Agent chưa đủ năng lực
        - Vung rao can tam ly   : Desire thấp, Capacity cao -> Agent làm được nhưng worker không muốn
                                   (đây là phát hiện phản biện quan trọng: rào cản không phải kỹ thuật)
        - Vung uu tien thap     : Desire thấp, Capacity thấp -> chưa cần ưu tiên

    Trả về df đã có thêm cột 'Zone', và in ra phân bố.
    """
    df = df.copy()
    d_med = df[desire_col].median()
    c_med = df[capacity_col].median()
    print(f"Ngưỡng trung vị (cấp task): Desire = {d_med:.2f}, Capacity = {c_med:.2f}")

    def _zone(row):
        d_high = row[desire_col] >= d_med
        c_high = row[capacity_col] >= c_med
        if d_high and c_high:
            return "Vung an toan"
        elif d_high and not c_high:
            return "Vung rui ro ky thuat"
        elif not d_high and c_high:
            return "Vung rao can tam ly"
        else:
            return "Vung uu tien thap"

    df["Zone"] = df.apply(_zone, axis=1)

    counts = df["Zone"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print("\nPhân bố 4 vùng Desire x Capacity:")
    for z in counts.index:
        print(f"  {z}: {counts[z]} ({pct[z]}%)")

    return df


def plot_desire_capacity_matrix(df, desire_col="Automation Desire Rating",
                                 capacity_col="Automation Capacity Rating",
                                 zone_col="Zone", save_path=None):
    """
    Vẽ ma trận tán xạ Desire vs Capacity, tô màu theo Zone.
    Yêu cầu df đã có cột zone_col (chạy classify_zone() trước).
    """
    if zone_col not in df.columns:
        raise ValueError(f"Thiếu cột '{zone_col}'. Hãy chạy classify_zone(df) trước.")

    d_med = df[desire_col].median()
    c_med = df[capacity_col].median()

    colors = {
        "Vung an toan": "#4C9A2A",
        "Vung rui ro ky thuat": "#2980B9",
        "Vung rao can tam ly": "#E0A800",
        "Vung uu tien thap": "#95A5A6",
    }

    fig, ax = plt.subplots(figsize=(7, 6))
    for z in df[zone_col].value_counts().index:
        sub = df[df[zone_col] == z]
        ax.scatter(sub[desire_col], sub[capacity_col], alpha=0.6, s=35,
                   color=colors.get(z, "gray"), label=z)

    ax.axvline(d_med, color="black", linestyle="--", linewidth=1)
    ax.axhline(c_med, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Automation Desire Rating (trung bình theo task)")
    ax.set_ylabel("Automation Capacity Rating (trung bình theo task)")
    ax.set_title("Đối chiếu Desire (worker) vs Capacity (expert) — cấp độ task")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")

    plt.show()
    return fig


def zone_summary(df, zone_col="Zone", reason_col="Dominant_Agency_Reason",
                  focus_zone="Vung rao can tam ly"):
    """
    Phân tích sâu vùng chỉ định (mặc định: Vùng rào cản tâm lý) — đây là vùng
    quan trọng nhất cho luận điểm phản biện "Capacity cao không đồng nghĩa nên
    triển khai Agent tự chủ". In ra danh sách task thuộc vùng này và lý do giữ
    quyền chiếm ưu thế.

    Trả về DataFrame con của vùng focus_zone, sort theo Capacity giảm dần.
    """
    sub = df[df[zone_col] == focus_zone].copy()
    sub = sub.sort_values("Automation Capacity Rating", ascending=False)

    print(f"Số task thuộc '{focus_zone}': {len(sub)}")
    print(f"\nPhân bố lý do giữ quyền chiếm ưu thế trong vùng này:")
    print(sub[reason_col].value_counts())

    print(f"\nTop 10 task có Capacity cao nhất nhưng Desire thấp (rào cản tâm lý rõ nhất):")
    display_cols = ["Occupation", "Automation Desire Rating", "Automation Capacity Rating", reason_col]
    print(sub[display_cols].head(10).to_string(index=False))

    return sub


REASON_TO_LEVEL = {
    "Control": ("Level 0-1: Chi goi y (suggestion-only)",
                "Rao can la TAM LY (muon tu lam), khong phai nang luc. "
                "Agent khong nen co gang tu chu hon; chi nen dua goi y, "
                "con nguoi quyet dinh va thuc hien."),
    "Domain Knowledge": ("Level 1-2: Ho tro co giai trinh (explainable assist)",
                          "Agent can co co che 'explain reasoning' va cho phep worker "
                          "chinh sua logic xu ly, khong chi sua output cuoi cung."),
    "Quality Oversight": ("Level 2-3: Tu chu co giam sat ky thuat",
                           "Agent co the hanh dong nhung can co confidence score / "
                           "audit log de worker kiem tra lai khi can, khong can giai trinh sau."),
    "Empathy": ("Level 1: Ho tro hau truong (backstage support)",
                "Task co yeu to con nguoi-voi-con nguoi; Agent chi nen ho tro chuan bi "
                "thong tin, khong thay the tuong tac truc tiep."),
    "Dynamic": ("Level 1-2: Ho tro linh hoat, can giam sat tinh huong",
                "Tinh chat task thay doi lien tuc; Agent can co kha nang escalate "
                "cho con nguoi khi gap tinh huong moi."),
    "Physical": ("Level 0-1: Ho tro mot phan, can con nguoi tai cho",
                 "Gioi han vat ly thuc te; Agent chi ho tro phan kha thi tu xa, "
                 "khong the thay the hoan toan."),
    "Ethical": ("Level 1: Ho tro co kiem duyet dao duc",
                "Can co lop kiem duyet/giai trinh ve mat dao duc truoc khi Agent hanh dong."),
}
 
 
def reason_to_autonomy_level(reason):
    """Tra ve (ten_level, mo_ta) tuong ung voi 1 loai ly do giu quyen."""
    return REASON_TO_LEVEL.get(reason, ("Level 1-2: Can xem xet them", "Ly do chua duoc dinh nghia san."))
 
 
def build_recommendation_table(top_reasons, task_df, occ_col="Occupation"):
    """
    Gộp kết quả occupation_top_reason() (3.3) với thống kê Desire/Capacity/Zone
    trung bình theo nghề (3.4), rồi gán mức độ tự chủ Agent khuyến nghị.
 
    Tham số:
        top_reasons : DataFrame, đầu ra của occupation_top_reason() trong analysis_3_3.py
                      (cột: Occupation, Top_Reason, Top_Reason_Pct, N)
        task_df     : DataFrame, đầu ra của classify_zone() trong analysis_3_4.py
                      (cần có cột Occupation, Automation Desire Rating, Automation Capacity Rating, Zone)
 
    Trả về DataFrame khuyến nghị, đã sort theo Top_Reason_Pct giảm dần.
    """
    occ_stats = task_df.groupby(occ_col).agg(
        Mean_Desire=("Automation Desire Rating", "mean"),
        Mean_Capacity=("Automation Capacity Rating", "mean"),
        N_Tasks=("Automation Desire Rating", "count"),
    ).round(2)
 
    rec = top_reasons.merge(occ_stats, left_on="Occupation", right_index=True, how="left")
 
    levels = rec["Top_Reason"].apply(reason_to_autonomy_level)
    rec["Autonomy_Level"] = levels.apply(lambda x: x[0])
    rec["Recommendation"] = levels.apply(lambda x: x[1])
 
    rec = rec.sort_values("Top_Reason_Pct", ascending=False).reset_index(drop=True)
 
    cols = ["Occupation", "Top_Reason", "Top_Reason_Pct", "Mean_Desire", "Mean_Capacity",
            "Autonomy_Level", "Recommendation"]
    print(rec[cols].to_string(index=False))
 
    return rec[cols]
 
 
def plot_autonomy_levels(rec_table, save_path=None):
    """
    Vẽ biểu đồ ngang thể hiện mức độ tự chủ Agent khuyến nghị cho từng nghề,
    màu theo loại lý do giữ quyền chiếm ưu thế.
    """
    level_order = [
        "Level 0-1: Chi goi y (suggestion-only)",
        "Level 0-1: Ho tro mot phan, can con nguoi tai cho",
        "Level 1: Ho tro hau truong (backstage support)",
        "Level 1: Ho tro co kiem duyet dao duc",
        "Level 1-2: Ho tro co giai trinh (explainable assist)",
        "Level 1-2: Ho tro linh hoat, can giam sat tinh huong",
        "Level 2-3: Tu chu co giam sat ky thuat",
    ]
    level_score = {lvl: i for i, lvl in enumerate(level_order)}
 
    df = rec_table.copy()
    df["Level_Score"] = df["Autonomy_Level"].map(level_score).fillna(-1)
    df = df.sort_values("Level_Score")
 
    reason_colors = {
        "Control": "#C0392B",
        "Domain Knowledge": "#E0A800",
        "Quality Oversight": "#2980B9",
        "Empathy": "#8E44AD",
        "Dynamic": "#16A085",
        "Physical": "#7F8C8D",
        "Ethical": "#D35400",
    }
    colors = df["Top_Reason"].map(reason_colors).fillna("gray")
 
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(df))))
    ax.barh(df["Occupation"], df["Level_Score"], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Occupation"])
    ax.set_xticks(range(len(level_order)))
    ax.set_xticklabels([f"L{i}" for i in range(len(level_order))])
    ax.set_xlabel("Mức độ tự chủ Agent khuyến nghị (thấp -> cao)")
    ax.set_title("Khuyến nghị mức độ tự chủ AI Agent theo nghề (ngành CS)")
 
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in reason_colors.values()]
    ax.legend(handles, reason_colors.keys(), title="Lý do giữ quyền chiếm ưu thế",
              loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Đã lưu biểu đồ: {save_path}")
 
    plt.show()
    return fig