# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phạm Đình Hải

**MSSV:** 2A202602482

**Khóa:** K4 - Track 3A

**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Với threshold 0.85 trên 26 tài liệu đọc được, semantic tạo 208 chunks, trung bình 99 ký tự; basic tạo 51 chunks, trung bình 410 ký tự. Threshold hiện khá cao nên tách mạnh, có chunk chỉ 6 ký tự; cần đặt thêm minimum chunk size trước production. |
| Hierarchical/structure chunking | M1 | `chunk_hierarchical()`, `chunk_structure_aware()` | Hierarchical tạo 11 parents và 110 children, child trung bình 189 ký tự, tối đa 256. Structure-aware tạo 106 chunks và giữ header/section metadata, phù hợp tài liệu policy dạng Markdown. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 xử lý exact terms/số liệu, dense xử lý tương đồng ngữ nghĩa, RRF hợp nhất mà không cần chuẩn hóa hai thang điểm. Pipeline dùng MiniLM fallback và index 117 chunks vào Qdrant; Context Precision tăng từ 0.6042 lên 0.9708. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Model lớn chưa có trong cache nên pipeline chuyển sang lexical fallback. Cold-start thất bại mất khoảng 14.98 giây; các lượt fallback sau dưới 1 ms. Các lỗi version cho thấy lexical overlap không thay thế được semantic cross-encoder. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Production đạt Faithfulness 0.8242, Answer Relevancy 0.8781, Context Precision 0.9708 và Context Recall 0.9000. Cả bốn đều trên 0.75; Faithfulness vẫn thấp nhất ở câu multi-hop và numeric reasoning. |
| Contextual enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | Combined mode làm giàu 117 chunks bằng một call/chunk. Kết hợp enrichment với retrieve-child/return-parent giúp Context Recall tăng từ 0.5750 lên 0.9000 và giữ rõ provenance theo tên tài liệu. |

## Phần 2: Khó khăn & Cách giải quyết

### Lỗi kỹ thuật gặp phải

- `PermissionError: [Errno 13] Permission denied ... ensurepip`: thư mục temp mặc định bị giới hạn quyền.
- `Failed to establish a new connection: [Errno 11001] getaddrinfo failed`: môi trường sandbox không cho pip truy cập mạng.
- `BAAI/bge-m3 does not appear to have a file named pytorch_model.bin or model.safetensors.`: model tải dở, không đủ weight trong cache.
- `We couldn't connect to 'https://huggingface.co' ... and couldn't find them in the cached files.`: reranker chưa được tải.
- `OPENAI_API_KEY is not configured`: lần chạy đầu M4 phải trả fallback 0.0; sau khi cấu hình key, RAGAS chạy đủ 80 jobs cho mỗi pipeline.

### Nguyên nhân và quá trình debug

1. Chuyển `TEMP`/`TMP` sang thư mục writable trong project để bootstrap pip cho `.venv`.
2. Cài dependencies với quyền mạng được cho phép, sau đó dùng `pip check` và import smoke test để xác nhận môi trường.
3. Khởi động Qdrant bằng Docker, kiểm tra cả REST endpoint lẫn `QdrantClient.get_collections()`.
4. Chạy model nhỏ ở offline mode để xác nhận cache. Khi bge-m3/reranker chưa đủ file, bổ sung fallback MiniLM và lexical reranking để pipeline vẫn chạy end-to-end.
5. Giữ metric fallback 0.0 ở lần chạy thiếu key, sau đó cấu hình API, chạy lại baseline/production và thay báo cáo bằng điểm thật.
6. Phân tích 8/10 failure trả “Không tìm thấy” dù context có đáp án; sửa pipeline để retrieve child nhưng trả parent, giữ tên nguồn và hướng dẫn LLM tổng hợp nhiều context. Kết quả cả bốn metric cuối đều vượt 0.75.

### Kiến thức còn thiếu & Cách khắc phục

- Cần tìm hiểu version-aware retrieval và metadata conflict resolution cho kho chính sách có tài liệu superseded.
- Cần benchmark cross-encoder thật sau khi tải đủ model, bao gồm cold start, warm latency và chất lượng top-3.
- Cần thực hành cấu hình RAGAS evaluator độc lập với generation model và quản lý chi phí/rate limit.

## Phần 3: Action Plan cho Project cá nhân

### Project: Trợ lý tra cứu chính sách nội bộ

#### 1. Hiện trạng

- **Pipeline hiện tại:** Hierarchical chunking → contextual enrichment → BM25 + dense → RRF → reranking → grounded answer → RAGAS.
- **Vấn đề:** Hai PDF scan chưa được OCR; bge-m3 và cross-encoder chưa cache đủ nên đang dùng fallback; câu mua laptop 30 triệu vẫn retrieval sai loại tài liệu; semantic threshold tạo một số chunk quá ngắn.

#### 2. Kế hoạch cải tiến

1. **Chunking:** Dùng structure-aware theo section, kết hợp parent-child; đặt child 256–400 ký tự và minimum 80 ký tự để tránh fragment quá ngắn.
2. **Search:** Dùng hybrid BM25 + bge-m3 + RRF; bổ sung metadata filter theo loại chính sách, năm hiệu lực và trạng thái hiện hành.
3. **Reranking:** Dùng `BAAI/bge-reranker-v2-m3` cho top-20 → top-3; giữ lexical fallback cho degraded mode.
4. **Evaluation:** Dùng RAGAS 4 metrics trên 20 câu hiện có, thêm custom metrics cho version correctness, negation accuracy, numeric exact match và latency.
5. **Enrichment:** Combined single-call để tạo summary, HyQA, context và metadata; mở rộng schema với `version`, `effective_date`, `status`, `supersedes`.
6. **Document ingestion:** Thêm OCR cho `BCTC.pdf` và Nghị định 13/2023, ghi rõ confidence và page provenance.

#### 3. Timeline triển khai

- **Tuần 1:** Hoàn thiện OCR, version metadata và chunk-quality checks.
- **Tuần 2:** Tải/benchmark bge-m3 cùng bge-reranker; tối ưu hybrid retrieval trên các câu version/numeric.
- **Tuần 3:** Thêm custom numeric/version metrics, xử lý bottom-5 và đặt regression gates trước khi deploy.
