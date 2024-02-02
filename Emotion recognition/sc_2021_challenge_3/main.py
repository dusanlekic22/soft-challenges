from process import extract_facial_expression_from_image, train_or_load_facial_expression_recognition_model
import glob
import sys
import os

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    TRAIN_DATASET_PATH = sys.argv[1]
else:
    TRAIN_DATASET_PATH = '.' + os.path.sep + 'dataset' + os.path.sep + 'train' + os.path.sep

if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[2]
else:
    VALIDATION_DATASET_PATH = '.'+os.path.sep+'dataset'+os.path.sep+'validation'+os.path.sep
# -------------------------------------------------------------------

# indeksiranje labela za brzu pretragu
label_dict = dict()
with open(TRAIN_DATASET_PATH+'annotations.csv', 'r') as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        if index > 0:
            cols = line.replace('\n', '').split(',')
            label_dict[cols[0]] = cols[1]

# priprema skupa podataka za metodu za treniranje
train_image_paths = []
train_image_labels = []
for image_name in os.listdir(TRAIN_DATASET_PATH):
    if '.jpg' in image_name:
        train_image_paths.append(os.path.join(TRAIN_DATASET_PATH, image_name))
        train_image_labels.append(label_dict[image_name])

# istrenirati model za prepoznavanje ekspresije lica
model = train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels)

# izvrsiti citanje teksta sa svih fotografija iz validacionog skupa podataka, koriscenjem istreniranog modela
processed_image_names = []
extracted_facial_expression = []

for image_path in glob.glob(VALIDATION_DATASET_PATH + "*.jpg"):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    extracted_facial_expression.append(extract_facial_expression_from_image(model, image_path))


# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima ekstrakcije za svaku sliku
result_file_contents = ""
for image_index, image_name in enumerate(processed_image_names):
    result_file_contents += "%s,%s\n" % (image_name, extracted_facial_expression[image_index])
# sacuvaj formirane rezultate u csv fajl
with open('result.csv', 'w') as output_file:
    output_file.write(result_file_contents)

# ------------------------------------------------------------------
