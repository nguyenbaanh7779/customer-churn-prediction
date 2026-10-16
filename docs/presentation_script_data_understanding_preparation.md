# Presentation Script — Data Understanding & Data Preparation

Thời lượng: **4 phút**, 5 slide chính (khoảng 530 từ tiếng Anh). Mỗi hàng trong bảng là một đoạn nói; cột trái để đọc khi thuyết trình, cột phải để hiểu nghĩa.

Mẹo: slide 3 dài nhất — nếu lố giờ, cắt đoạn cuối của slide 3 ("The same analysis gives us a second threshold…") trước. Số đọc dạng làm tròn (73%, 97%, 83%); số chính xác nằm trên slide.

## Slide 1 — From transactions to customer data (≈0:40)

| English (để nói) | Tiếng Việt (để hiểu) |
|---|---|
| The goal of data understanding and preparation is simple: understand the data, find insights about transactions and customers, and turn them into a strategy for building our dataset — the population, the label and the features. | Mục tiêu của bước hiểu dữ liệu và chuẩn bị dữ liệu rất đơn giản: hiểu dữ liệu, tìm ra insight về giao dịch và khách hàng, rồi chuyển chúng thành chiến lược xây dựng bộ dữ liệu — population, label và feature. |
| We start with about **1.07 million transaction rows**. After cleaning — mainly removing rows with no customer ID, plus duplicates, cancellations and invalid prices — **73% remain**: about **36,600 orders from 5,852 customers**, over two years, from December 2009 to December 2011. | Ban đầu có khoảng **1,07 triệu dòng giao dịch**. Sau khi làm sạch — chủ yếu loại các dòng không có mã khách hàng, cùng với dòng trùng, đơn hủy và giá không hợp lệ — còn lại **73%**: khoảng **36.600 đơn hàng của 5.852 khách**, trong hai năm, từ tháng 12/2009 đến tháng 12/2011. |

## Slide 2 — Seasonal sales; repeat customers drive revenue (≈0:50)

| English (để nói) | Tiếng Việt (để hiểu) |
|---|---|
| In a typical month, around **1,000 customers** buy, generating about **595 thousand pounds** in revenue. Sales are clearly seasonal: they peak in **October and November**, when customers order ahead of Christmas, and then drop sharply in December. | Một tháng điển hình có khoảng **1.000 khách** mua hàng, tạo ra khoảng **595 nghìn bảng Anh** doanh thu. Doanh số có tính mùa vụ rõ rệt: đạt đỉnh vào **tháng 10 và tháng 11**, khi khách đặt hàng trước Giáng sinh, rồi giảm mạnh vào tháng 12. |
| More importantly, **72% of customers are repeat buyers**, and they generate **97% of all revenue**. One-time buyers are more than a quarter of customers, but only about 3% of revenue. So almost all revenue comes from existing customers — and that is exactly why we need a strategy to keep them. | Quan trọng hơn, **72% khách là khách mua lặp lại**, và họ tạo ra **97% tổng doanh thu**. Khách chỉ mua một lần chiếm hơn một phần tư số khách nhưng chỉ khoảng 3% doanh thu. Như vậy gần như toàn bộ doanh thu đến từ khách cũ — và đó chính là lý do cần có chiến lược giữ chân họ. |

## Slide 3 — Churn is inferred from purchase rhythm → 90 days (≈1:00)

| English (để nói) | Tiếng Việt (để hiểu) |
|---|---|
| But in retail, customers never cancel a contract. They simply stop buying. So we have to define churn as **no purchase within a certain number of days** — and the question is: how many? | Nhưng trong bán lẻ, khách không bao giờ hủy hợp đồng. Họ chỉ đơn giản là ngừng mua. Vì vậy ta phải định nghĩa churn là **không mua hàng trong một số ngày nhất định** — và câu hỏi là: bao nhiêu ngày? |
| We looked at the time between consecutive purchases. **55% of repeat purchases happen within 30 days** — the biggest group, but not the full picture. Within **90 days, we cover 83%** of repeat purchases. | Chúng tôi xem khoảng thời gian giữa hai lần mua liên tiếp. **55% lần mua lại xảy ra trong vòng 30 ngày** — nhóm lớn nhất nhưng chưa phải toàn cảnh. Trong vòng **90 ngày thì bao phủ được 83%** số lần mua lại. |
| This is a trade-off. A shorter window labels too many customers as churned who would actually come back. A longer window detects churn too late, and on two years of data it leaves too little training data: 120 days would add only 5 points of coverage, but cut our training weeks from 24 to just 10. So **90 days is the balance point**. | Đây là một sự đánh đổi. Cửa sổ ngắn hơn sẽ gắn nhãn churn cho quá nhiều khách mà thực ra họ vẫn quay lại. Cửa sổ dài hơn thì phát hiện churn quá muộn, và với hai năm dữ liệu thì còn quá ít dữ liệu huấn luyện: 120 ngày chỉ tăng thêm 5 điểm độ bao phủ nhưng làm số tuần huấn luyện giảm từ 24 xuống còn 10. Vì vậy **90 ngày là điểm cân bằng**. |
| The same analysis gives us a second threshold: customers who haven't bought for more than **136 days** are treated as already lost, so we don't score them. | Cùng phân tích này cho ra ngưỡng thứ hai: khách đã không mua quá **136 ngày** được coi là đã mất, nên không chấm điểm họ nữa. |

## Slide 4 — 82% of revenue comes from customers who return within 90 days (≈0:40)

| English (để nói) | Tiếng Việt (để hiểu) |
|---|---|
| When we group customers by their usual buying rhythm, those who typically return within 90 days generate **82% of revenue**. | Khi nhóm khách theo nhịp mua thường lệ, những khách thường quay lại trong vòng 90 ngày tạo ra **82% doanh thu**. |
| So when one of these customers goes 90 days without buying, that's a clear break from their normal pattern — and it's exactly the signal our 90-day label captures, on the customers who matter most. | Vì vậy, khi một khách thuộc nhóm này đi qua 90 ngày mà không mua, đó là sự lệch rõ ràng khỏi thói quen bình thường của họ — và đó chính là tín hiệu mà label 90 ngày nắm bắt, trên đúng nhóm khách quan trọng nhất. |

## Slide 5 — From insights to the training dataset (≈0:50)

| English (để nói) | Tiếng Việt (để hiểu) |
|---|---|
| Based on these insights, we built the dataset. Each row is **one customer at one weekly snapshot**. Every week, we take all active customers — about **2,600 per week**, or 209 thousand rows in total. | Dựa trên các insight này, chúng tôi xây dựng bộ dữ liệu. Mỗi dòng là **một khách hàng tại một thời điểm snapshot hàng tuần**. Mỗi tuần lấy toàn bộ khách còn active — khoảng **2.600 khách mỗi tuần**, tổng cộng 209 nghìn dòng. |
| The label is churn if the customer makes **no purchase in the next 90 days** — and the churn rate is **48%**, so churn is not a rare event here. | Label là churn nếu khách **không mua hàng trong 90 ngày tiếp theo** — tỷ lệ churn là **48%**, nên ở đây churn không phải là sự kiện hiếm. |
| We built **11 features** — recency, frequency, monetary value, purchase gaps, product variety and cancellations — using only information available at the snapshot date, to avoid leakage. | Chúng tôi xây dựng **11 feature** — recency, frequency, giá trị chi tiêu, khoảng cách giữa các lần mua, mức độ đa dạng sản phẩm và số lần hủy đơn — chỉ dùng thông tin có tại thời điểm snapshot, để tránh rò rỉ dữ liệu (leakage). |
| Finally, we split the data **by time**, with a **90-day gap** between train, validation and test, so the model is always evaluated on a truly unseen future period. | Cuối cùng, chúng tôi chia dữ liệu **theo thời gian**, với **khoảng đệm 90 ngày** giữa các tập train, validation và test, để model luôn được đánh giá trên một giai đoạn tương lai thực sự chưa từng thấy. |
| With that, the dataset is ready for modeling. | Như vậy, bộ dữ liệu đã sẵn sàng cho bước xây dựng model. |

## Từ vựng cần chú ý

| English | Tiếng Việt |
|---|---|
| repeat buyers / repeat purchases | khách mua lặp lại / lần mua lại |
| consecutive purchases | hai lần mua liên tiếp |
| coverage | độ bao phủ |
| trade-off | sự đánh đổi |
| balance point | điểm cân bằng |
| break from their normal pattern | lệch khỏi thói quen bình thường |
| snapshot | thời điểm chụp trạng thái khách hàng |
| leakage | rò rỉ thông tin tương lai vào feature |
| truly unseen future period | giai đoạn tương lai mà model chưa từng thấy |
