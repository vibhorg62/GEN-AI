from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)
#Schema

class Review(TypedDict):
    
    key_themes:Annotated[list[str], "A list of the main themes or topics discussed in the review."]
    summary:Annotated[str, "A brief summary of the review."]
    sentiment:Annotated[Literal["mazedaar jhakkaas", "ganda ekdum", "noormaal"], "The overall sentiment of the review, e.g., positive, negative, neutral."]
    pros:Annotated[Optional[list[str]], "A list of positive aspects or advantages mentioned in the review."]
    cons:Annotated[Optional[list[str]], "A list of negative aspects or disadvantages mentioned in the review."]
    
structured_model=model.with_structured_output(Review)

result = structured_model.invoke("""I have been using this smartphone for nearly six months, and I can confidently say that it has exceeded my expectations in almost every aspect. Right from the first day, the phone felt premium in hand because of its excellent build quality and sleek design. The display is absolutely gorgeous, with vibrant colors, deep contrast, and a smooth high refresh rate that makes scrolling, gaming, and watching videos incredibly enjoyable. Even under bright sunlight, the screen remains clear and easy to read.

Performance has been outstanding throughout my usage. Whether I'm multitasking between several applications, editing photos, attending video meetings, or playing graphics-intensive games, the phone handles everything effortlessly without any noticeable lag. Apps launch almost instantly, switching between tasks is seamless, and the overall experience feels fast and responsive. I was particularly impressed by how well it manages heavy workloads while maintaining consistent performance.

Battery life is one of the biggest highlights of this device. Even with heavy daily usage that includes gaming, streaming videos, browsing social media, GPS navigation, and taking photos, the battery comfortably lasts an entire day. The fast charging technology is incredibly convenient, allowing the phone to charge from nearly empty to a high percentage in a very short amount of time. This has been especially helpful during busy days when I don't have much time to wait for charging.

The camera system is another area where this phone truly shines. Daylight photos are exceptionally sharp, colorful, and full of detail. Portrait mode produces natural-looking background blur, while landscape shots capture impressive dynamic range. Even night mode performs surprisingly well, delivering bright and detailed images with minimal noise. Video recording is smooth, stabilization works effectively, and the audio quality in videos is clear. The front camera also takes excellent selfies with accurate skin tones and plenty of detail.

The software experience has been clean, intuitive, and reliable. The interface feels modern, animations are smooth, and I haven't experienced any major bugs or crashes. The company has also released regular software updates that not only introduce useful new features but also improve security and optimize performance. I appreciate that the phone comes with very little unnecessary software, making the overall user experience feel lightweight and polished.

Connectivity has been excellent as well. Wi-Fi, Bluetooth, GPS, and mobile network performance have all been consistently reliable. Call quality is crystal clear, and the speakers produce rich, immersive sound whether I'm watching movies or listening to music. The fingerprint sensor is extremely fast and accurate, while face unlock works instantly under most lighting conditions.

Customer service deserves appreciation too. I contacted support once with a small question regarding device settings, and the response was quick, professional, and genuinely helpful. It's reassuring to know that the company provides quality after-sales support.

Overall, this smartphone offers an exceptional combination of premium design, powerful performance, excellent cameras, long battery life, reliable software, and great customer support. Considering its price, it delivers outstanding value for money and easily competes with much more expensive flagship devices. I would highly recommend this phone to students, working professionals, gamers, and anyone looking for a dependable smartphone that performs exceptionally well in every situation. It has been one of the best smartphone purchases I have made, and I am extremely satisfied with my experience.""")

print(result)
print(result['sentiment'])