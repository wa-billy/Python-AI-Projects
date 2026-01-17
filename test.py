import tensorflow as tf
from keras.layers import TextVectorization

# 1. กำหนดค่าเลเยอร์
vectorize_layer = TextVectorization(
    max_tokens=10000,           # จำนวนคำสูงสุดในพจนานุกรม
    output_mode='int',          # ส่งออกเป็นเลขจำนวนเต็ม (Index)
    output_sequence_length=100,  # กำหนดความยาวประโยคให้เท่ากัน (Padding/Truncating)
    
)

# 2. สอนให้เลเยอร์รู้จักคำศัพท์ (Adapt)
# ต้องใช้ข้อมูลตัวอย่าง (Corpus) เพื่อสร้าง Vocabulary
train_text = ["ฉันรักการเรียนรู้", "Keras ใช้งานง่ายมาก", "การทำ NLP สนุกดี"]
vectorize_layer.adapt(train_text)

# 3. ลองใช้งานแปลงข้อความ
sample_input = ["ฉันรัก Keras"]
vector_output = vectorize_layer(sample_input)
print(len(vector_output))
