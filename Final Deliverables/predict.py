import numpy as np
from keras.preprocessing import image 
from keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from keras.models import Sequential
from tensorflow.keras.utils import img_to_array, array_to_img, load_img

json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("final_model.h5")

img = load_img('fre.jpg', target_size=(224, 224))
        
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
#x = np.array(img)
preds = loaded_model.predict([x.reshape(1, 224,224,3)])
ClassIndex = np.argmax(preds,axis=1)

found = ["The great Indian bustard is a bustard found on the Indian subcontinent. A large bird with a horizontal body and long bare legs, giving it an ostrich like appearance, this bird is among the heaviest of the flying birds. It belongs to Otididae family and is listed among critically endangered species.",
                 "The spoon-billed sandpiper is a small wader which breeds in northeastern Russia and winters in Southeast Asia. It belongs to Scolopacidae family and is listed among critically endangered species.",
                 "Corpse Flower or Amorphophallus Titanum is endemic to sumantra. Due to its odor, like that of a rotting corpse, the titan arum is characterized as a Carrion Flower or Corpse Flower. It belongs to Araceae family.",
                 "Lady's slipper, (subfamily Cypripedioideae), also called lady slipper or slipper orchid, subfamily of five genera of orchids (family Orchidaceae), in which the lip of the flower is slipper-shaped.",
                 "Pangolins, sometimes known as scaly anteaters, are of the order Pholidota. Often thought of as a reptile, but pangolins are actually mammals. They are the most trafficked mammals.",
                 "The white deer found at Seneca Army Depot are a natural variation of the white-tailed deer (Odocoileus virginianus), which usually have brown coloring. The Seneca White Deer are leucistic, meaning they lack all pigmentation in the hair, but have the normal brown-colored eyes."
        ]
print(found[ClassIndex[0]])

