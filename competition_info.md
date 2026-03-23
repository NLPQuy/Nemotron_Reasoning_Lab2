# NVIDIA Nemotron Model Reasoning Challenge

**Khám phá và nâng cao kỹ thuật suy luận sử dụng mô hình mở NVIDIA Nemotron trên một bộ tiêu chuẩn (benchmark) mới.**

## Tổng quan
Cuộc thi tập trung vào việc phát triển các kỹ thuật giúp cải thiện độ chính xác trong khả năng suy luận của các mô hình NVIDIA Nemotron. Người tham gia sẽ thử nghiệm với việc thiết kế câu lệnh (prompting), xử lý luồng dữ liệu (data pipelines), và tinh chỉnh nhẹ (lightweight fine-tuning) trong khi đánh giá các phương pháp của họ trên một bộ chuẩn suy luận hoàn toàn mới do NVIDIA Research phát triển.

## Mô tả chi tiết
Các bộ chuẩn suy luận là một công cụ hữu ích để đo lường tiến bộ của các mô hình trong các tác vụ có cấu trúc. Hiện nay, các cải tiến về khả năng suy luận thường được khám phá thông qua nhiều nỗ lực độc lập, sử dụng nhiều tập dữ liệu, prompt và thiết lập đánh giá khác nhau, khiến việc so sánh trực tiếp trở nên khó khăn. Một bộ chuẩn chung kết hợp cùng mô hình cơ sở thống nhất sẽ cho phép các kỹ thuật được kiểm thử và so sánh nhất quán hơn.

Trong cuộc thi này, người tham gia sẽ làm việc với mô hình cơ sở mở **Nemotron 3 Nano** (cụ thể là `Nemotron-3-Nano-30B`) và bộ chuẩn của NVIDIA Research. Bạn có thể tự do thử nghiệm với:
- Các chiến lược Prompting
- Lọc và tinh tuyển dữ liệu (Data filtering and curation)
- Tạo dữ liệu tổng hợp (Synthetic data generation)
- Học tăng cường (Reinforcement learning)
- Tinh chỉnh nhẹ (Lightweight fine-tuning)
- Hoặc các phương pháp khác do bạn chọn.

Người tham gia có thể sử dụng bất kỳ framework huấn luyện hoặc thư viện nào (như Hugging Face, Unsloth, Axolotl, TRL,...). Yêu cầu duy nhất là bài nộp cuối cùng phải tạo ra một LoRA adapter tương thích với mô hình cơ sở. Các giải pháp đòi hỏi phải có tài liệu minh bạch và rõ ràng (Notebook và báo cáo) để thúc đẩy khả năng tái tạo và học hỏi chung của cộng đồng.

## Tiêu chí Đánh giá (Evaluation)
Các bài nộp được đánh giá dựa trên **Độ chính xác (Accuracy)**. 

Mô hình NVIDIA Nemotron-3-Nano-30B sẽ được tải kèm với adapter LoRA của bạn (bắt buộc phải có tệp `adapter_config.json`) thông qua engine suy luận vLLM. Trong mỗi trường hợp kiểm thử, mô hình phải tạo ra câu trả lời cuối cùng và đặt nó bên trong lệnh LaTeX `\boxed{}`. Thước đo sẽ trích xuất đáp án nằm trong hộp (hoặc sử dụng các phương pháp tìm kiếm heuristic / số cuối cùng nếu không có hộp). Một dự đoán được coi là chính xác nếu khớp với đáp án đúng chính xác theo dạng chuỗi hoặc có sai số tương đối trong khoảng $10^{-2}$. Điểm số cuối cùng là tỷ lệ các câu trả lời đúng.

**Các thông số chạy inference:**
- `max_lora_rank`: 32
- `max_tokens`: 7680
- `top_p`: 1.0
- `temperature`: 0.0
- `max_num_seqs`: 64
- `gpu_memory_utilization`: 0.85
- `max_model_len`: 8192

**Quy cách nộp bài:**
Bạn phải nộp một adapter LoRA với rank tối đa là 32, đóng gói gọn trong tệp `submission.zip`.

## Cột mốc thời gian (Timeline)
*(Tất cả thời hạn kết thúc lúc 11:59 PM UTC)*

- **16 tháng 3, 2026**: Ngày bắt đầu.
- **9 tháng 4, 2026**: Ngày chốt điểm giữa kỳ (Midpoint Cut-off Date).
- **8 tháng 6, 2026**: Hạn chót chấp nhận nội quy và tham gia cuộc thi.
- **8 tháng 6, 2026**: Hạn chót hợp nhất đội tuyển.
- **15 tháng 6, 2026**: Ngày kết thúc cuộc thi & Hạn nộp bài chung cuộc.

## Giải thưởng (Prizes)
Để đủ điều kiện nhận thưởng, các đội phải công khai Kaggle Notebook và bài viết (write-up) giải thích chi tiết phương pháp, dữ liệu và kỹ thuật của mình.

**Giải Bảng xếp hạng Chung cuộc:**
- **Hạng 1:** $25,000 + 5 hệ thống DGX Sparks
- **Hạng 2:** $15,000 + 2 hệ thống DGX Sparks
- **Hạng 3:** $5,000 + 1 hệ thống DGX Sparks

*(Lưu ý: Có tổng cộng 8 hệ thống NVIDIA DGX Spark trị giá ~$4,699/hệ thống được trao thưởng cho các cá nhân trong đội thắng cuộc đạt đủ điều kiện).*

**Giải Tiến bộ Mở (Mid-Competition Milestone):**
- **Trị giá:** $5,000 + 1 hệ thống DGX Spark
- Trao cho đội đứng đầu bảng xếp hạng tính đến ngày chốt điểm giữa kỳ (9/4/2026) và nộp tài liệu giải pháp hợp lệ trước 16/4/2026.

**Giải Đóng góp Mở (Open Contribution Awards):**
- Dành cho top 10% các đội trên bảng xếp hạng có kỹ thuật mang tính đột phá:
  - Phương pháp Xử lý Dữ liệu/Dữ liệu tổng hợp tốt nhất: 1 DGX Spark
  - Phương pháp Học tăng cường (RL) tốt nhất: 1 DGX Spark
  - Phương pháp Fine-tuning tốt nhất: 1 DGX Spark

## Năng lực tính toán (Compute)
Cuộc thi hợp tác với Google Cloud để cung cấp hệ thống máy ảo G4 chạy bằng GPU **NVIDIA RTX PRO 6000 Blackwell Server Edition**, mang đến hiệu suất và bộ nhớ cần thiết để fine-tuning và chạy suy luận dữ liệu (inference) hiệu quả cao.