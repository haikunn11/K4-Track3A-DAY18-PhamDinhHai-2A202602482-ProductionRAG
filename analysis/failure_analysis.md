# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phạm Đình Hải

**MSSV:** 2A202602482

**Khóa:** K4 - Track 3A

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|--:|
| Faithfulness | 0.5208 | 0.8242 | +0.3033 |
| Answer Relevancy | 0.4810 | 0.8781 | +0.3971 |
| Context Precision | 0.6042 | 0.9708 | +0.3667 |
| Context Recall | 0.5750 | 0.9000 | +0.3250 |

Production đạt trên 0.75 ở cả bốn metric. Cải thiện lớn nhất là Answer Relevancy và Context Precision. Faithfulness thấp nhất, chủ yếu ở câu multi-hop hoặc cần phép tính từ nhiều tài liệu.

## Bottom-5 Failures

### #1 — Mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Director phê duyệt; cần xác nhận cấu hình từ CNTT và ít nhất 3 báo giá.
- **Got:** Trưởng phòng phê duyệt; CNTT cung cấp thông tin yêu cầu/tính năng.
- **Worst metric:** Faithfulness — 0.0000.
- **Error Tree:** Output sai → Context đúng? **Không** → Query OK? **Có** → Retrieval bị hút bởi số “30 triệu” trong chính sách hoàn chi đào tạo.
- **Root cause:** Top-3 gồm đào tạo, đào tạo nội bộ và nghỉ phép; không có `mua_sam.md`. Dense fallback chưa phân biệt ý định “mua laptop” với các chính sách có cùng số tiền.
- **Suggested fix:** Boost exact term “mua/mua sắm/laptop/thiết bị CNTT” trong BM25; thêm category filter `procurement`; tạo HyQA riêng cho ngưỡng phê duyệt và yêu cầu báo giá.

### #2 — Nghỉ không lương 20 ngày

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** CEO phê duyệt; nghỉ trên 14 ngày phải tự đóng phần bảo hiểm của nhân viên.
- **Got:** Trả đúng CEO nhưng bỏ sót lưu ý bảo hiểm.
- **Worst metric:** Faithfulness — 0.5000.
- **Error Tree:** Output chưa đủ → Context đúng? **Có ở context #2** → Query OK? **Có** → Answer generation chỉ lấy điều kiện phê duyệt trực tiếp, bỏ qualifier liên quan.
- **Root cause:** Prompt ưu tiên câu trả lời ngắn nên chưa yêu cầu nêu các hệ quả bắt buộc đi kèm khoảng thời gian nghỉ.
- **Suggested fix:** Yêu cầu LLM trả cả “người phê duyệt” và “điều kiện/hệ quả bắt buộc”; thêm checklist extraction cho policy có nhiều rule.

### #3 — Lương thử việc Junior

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** 17.000.000 VNĐ, kèm phép tính đúng.
- **Worst metric:** Faithfulness — 0.2500.
- **Error Tree:** Output đúng → Context đúng? **Có, từ `thu_viec.md` và `bang_luong_2024.md`** → Query OK? **Có** → RAGAS faithfulness đánh giá thấp claim suy diễn từ phép tính/liên kết hai nguồn.
- **Root cause:** Đáp án tổng hợp hai context và tạo claim “Junior P1-P2” cùng phép nhân; các claim suy diễn có thể bị judge coi là không được hỗ trợ trực tiếp.
- **Suggested fix:** Trả lời ngắn kèm provenance: “Bảng lương: tối đa 20 triệu; chính sách thử việc: 85%; kết quả: 17 triệu”; bổ sung custom numeric exact-match metric.

### #4 — Hoàn chi khóa học

- **Question:** Được tài trợ 25 triệu, nghỉ sau 8 tháng thì hoàn trả bao nhiêu?
- **Expected:** Hoàn 100%, tức 25.000.000 VNĐ.
- **Got:** Trả đúng 25 triệu và giải thích chưa đủ cam kết 1 năm.
- **Worst metric:** Faithfulness — 0.3333.
- **Error Tree:** Output đúng → Context đúng? **Có ở context #1** → Query OK? **Có** → Judge phạt các câu diễn giải lặp lại dù kết luận được hỗ trợ.
- **Root cause:** Answer dài hơn cần thiết và biến dữ kiện từ câu hỏi thành nhiều factual claims.
- **Suggested fix:** Giới hạn answer thành rule + phép áp dụng + kết quả, tránh lặp dữ kiện đầu vào như claim mới.

### #5 — Phạt tạm ứng quá hạn

- **Question:** Tạm ứng 15 triệu, thanh toán sau 20 ngày, bị phạt bao nhiêu?
- **Expected:** Trễ 5 ngày; 2%/tháng, pro-rata khoảng 50.000 VNĐ.
- **Got:** Trả đúng 50.000 VNĐ với phép tính đầy đủ.
- **Worst metric:** Faithfulness — 0.4000.
- **Error Tree:** Output đúng → Context đúng? **Có ở `tam_ung.md`** → Query OK? **Có** → Metric faithfulness nhạy với kết quả số được suy ra thay vì xuất hiện nguyên văn.
- **Root cause:** Context nêu 2%/tháng nhưng không nêu rõ công thức pro-rata theo 30 ngày; LLM tự áp dụng quy ước hợp lý.
- **Suggested fix:** Dùng calculator/tool rule rõ ràng và lưu calculation trace; thêm metric numeric exact match để bổ sung cho RAGAS.

## Case Study

**Question chọn phân tích:** Mua laptop 30 triệu cho nhân viên mới.

**Error Tree walkthrough:**

1. **Output đúng?** Không: sai cấp phê duyệt và thiếu 3 báo giá.
2. **Context đúng?** Không: không có chính sách mua sắm trong top-3.
3. **Query rewrite OK?** Chưa: chưa chuẩn hóa thành “mua sắm thiết bị CNTT 30 triệu”.
4. **Fix ở bước:** Query enrichment → category-aware hybrid retrieval → reranking.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Thêm metadata category `procurement`, `training`, `leave`, `security`.
- Tăng trọng số BM25 cho danh từ ý định và giảm tác động của số tiền xuất hiện ở tài liệu sai loại.
- Thêm regression test cho câu laptop 30 triệu.
- Dùng custom numeric metric cho ba câu cần tính toán để tránh phụ thuộc hoàn toàn vào LLM judge.
