import gradio as gr
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
from tensorflow.keras import mixed_precision

# Load your trained models
model = from_pretrained_keras("ml-debi/EfficientNetV2S-StanfordDogsA")

# Add information about the models
model_info = """
## 🐾 **Welcome to My Dog Breed Classifier!** 🐾

Unleash the power of AI and step into the world of dog breeds with my image classification app. This isn't just any app, it's your personal guide to the canine kingdom. 

My app uses the state-of-the-art **EfficientNetV2s model**, fine-tuned on the renowned **Stanford Dogs dataset**. This means you're getting top-tier, accurate results every time you use my app. 

Ever wondered what breed that adorable pooch in the park is? Or maybe you're curious about the lineage of your own furry friend? Snap a picture, upload it to my app, and voila! You'll have your answer in no time. 

From Affenpinschers to Yorkshire Terriers and everything in between, my app classifies over 120 breeds with ease. So why wait? Dive into the diverse world of dog breeds and discover something new today!

**Join me on this pawsome adventure!** 🐶

"""

examples = [["./examples/border_collie.jpg"], ["./examples/German-Shepherd.jpg"], ["./examples/staffordshire-bull-terrier-puppy.jpg"]]

def preprocess(image):
    print("before resize", image.shape)
    image = tf.image.resize(image, [224, 224])
    
    image = tf.expand_dims(image, axis=0)
    print("After expanddims", image.shape)
    return image

def predict(image):

    if mixed_precision.global_policy() == "mixed_float16":
        mixed_precision.set_global_policy(policy="float32")

    image = preprocess(image)
    print(mixed_precision.global_policy())
    prediction = model.predict(image)[0]
    print("model prediction", prediction)
    confidences = {model.config['id2label'][str(i)]: float(prediction[i]) for i in range(101)}
    return confidences

iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image()],
    outputs=[gr.Label(num_top_classes=5)],
    title="Dog Vision Mini Project",
    description=f"{model_info}\n",
    examples=examples
)

iface.launch()