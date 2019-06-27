from google.cloud import vision
import io
from pprint import pprint as pp
import json
import os

from skimage.io import imread, imshow
from skimage.transform import rescale

from google.cloud import vision_v1
from google.cloud.vision_v1 import enums
import six
import matplotlib.pyplot as plt


def sample_async_batch_annotate_images(input_image_uri, output_uri):
  """Perform async batch image annotation"""

  client = vision_v1.ImageAnnotatorClient()

  # input_image_uri = 'gs://cloud-samples-data/vision/label/wakeupcat.jpg'
  # output_uri = 'gs://your-bucket/prefix/'

  if isinstance(input_image_uri, six.binary_type):
    input_image_uri = input_image_uri.decode('utf-8')
  if isinstance(output_uri, six.binary_type):
    output_uri = output_uri.decode('utf-8')
  source = {'image_uri': input_image_uri}
  image = {'source': source}
  type_ = enums.Feature.Type.LABEL_DETECTION
  features_element = {'type': type_}
  type_2 = enums.Feature.Type.IMAGE_PROPERTIES
  features_element_2 = {'type': type_2}
  type_3 = enums.Feature.Type.OBJECT_LOCALIZATION
  features_element_3 = {'type': type_3}

  features = [features_element, features_element_2, features_element_3]
  requests_element = {'image': image, 'features': features}
  requests = [requests_element]
  gcs_destination = {'uri': output_uri}

  # The max number of responses to output in each JSON file
  batch_size = 2
  output_config = {'gcs_destination': gcs_destination, 'batch_size': batch_size}

  operation = client.async_batch_annotate_images(requests, output_config)

  print('Waiting for operation to complete...')
  response = operation.result()

  # The output is written to GCS with the provided output_uri as prefix
  gcs_output_uri = response.output_config.gcs_destination.uri
  print('Output written to GCS with prefix: {}'.format(gcs_output_uri))


def jsonfy(obj):
    labels = obj.label_annotations
    label_dicts = []  # Array that will contain all the EntityAnnotation dictionaries

    for label in labels:
        # Write each label (EntityAnnotation) into a dictionary
        dict = {'description': label.description, 'score': label.score, 'mid': label.mid, 'topicality': label.topicality}

        # Populate the array
        label_dicts.append(dict)

    colors = obj.image_properties_annotation.dominant_colors.colors
    colors_dicts = []
    for color_desc in colors:
        dict = {"color": {"red": color_desc.color.red,
                         "green": color_desc.color.green,
                         "blue": color_desc.color.blue},
                "score": color_desc.score,
                "pixelFraction": color_desc.pixel_fraction}
        colors_dicts.append(dict)
    properties_dict = {'dominantColors': { 'colors' : colors_dicts } }

    local_objects_dicts = []
    for local_obj in obj.localized_object_annotations:
        dict = {"mid": local_obj.mid,
                "name": local_obj.name,
                "score": local_obj.score,
                "boundingPoly": {
                    "normalizedVertices": [
                        {'x': vert.x, 'y': vert.y} for vert in local_obj.bounding_poly.normalized_vertices
                    ]

                    }
                }
        local_objects_dicts.append(dict)


    json_response = {
        'labelAnnotations' : label_dicts,
        'imagePropertiesAnnotation' : properties_dict,
        'localizedObjectAnnotations' : local_objects_dicts
    }

    return json.dumps(json_response)

def build_request(path):

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    # source = {'image_uri': input_image_uri}
    # image = {'source': source}
    type_ = enums.Feature.Type.LABEL_DETECTION
    features_element = {'type': type_}
    type_2 = enums.Feature.Type.IMAGE_PROPERTIES
    features_element_2 = {'type': type_2}
    type_3 = enums.Feature.Type.OBJECT_LOCALIZATION
    features_element_3 = {'type': type_3}

    features = [features_element, features_element_2, features_element_3]
    requests_element = {'image': image, 'features': features}
    return requests_element

def build_request_uri(bucket, name):
    # 'gs://cloud-samples-data/vision/label/wakeupcat.jpg'
    input_image_uri = "gs://{}/{}".format(bucket, name)
    source = {'image_uri': input_image_uri}
    image = {'source': source}
    type_ = enums.Feature.Type.LABEL_DETECTION
    features_element = {'type': type_}
    type_2 = enums.Feature.Type.IMAGE_PROPERTIES
    features_element_2 = {'type': type_2}
    type_3 = enums.Feature.Type.OBJECT_LOCALIZATION
    features_element_3 = {'type': type_3}

    features = [features_element, features_element_2, features_element_3]
    requests_element = {'image': image, 'features': features}
    return requests_element


def annanotate_image(data_dir, output_uri):
    client = vision.ImageAnnotatorClient()

    requests = [build_request_uri("michelin-data/data", name) for name in os.listdir(data_dir)]

    # requests = [build_request(path) for path in [os.path.join(data_dir, name) for name in os.listdir(data_dir)]]
    # pp(requests)
    # print (len(requests))

    gcs_destination = {'uri': output_uri}

    # The max number of responses to output in each JSON file
    batch_size = 2
    output_config = {'gcs_destination': gcs_destination, 'batch_size': batch_size}

    operation = client.async_batch_annotate_images(requests, output_config)
    print('Waiting for operation to complete...')
    response = operation.result()
    pp(response)

    print('Writing to file...')

    with open("response.txt", 'w') as f:
        f.write(str(response))



def annanotate_image_local(data_dir, output_dir, lim=None):
    client = vision.ImageAnnotatorClient()

    names = os.listdir(data_dir)
    if lim is not None:
        names = names[:lim]

    requests = [build_request(im_path) for im_path in [os.path.join(data_dir, name) for name in names]]

    # The max number of responses to output in each JSON file
    batch_size = 50
    # output_config = {'gcs_destination': gcs_destination, 'batch_size': batch_size}
    pos = 0
    for name, request in zip(names, requests):
        try:
            response = client.annotate_image(request)
            with open(os.path.join(output_dir, name[:-4] + ".json"), 'w') as f:
                f.write(jsonfy(response))
        except Exception as e:
            print(e)
            print(name)
        pos += 1
        if pos % 100 == 0:

            print(pos)



def detect_properties(path):
    """Detects image properties in the file."""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    print('Properties:')

    for color in props.dominant_colors.colors:
        print('fraction: {}'.format(color.pixel_fraction))
        print('\tr: {}'.format(color.color.red))
        print('\tg: {}'.format(color.color.green))
        print('\tb: {}'.format(color.color.blue))
        print('\ta: {}'.format(color.color.alpha))

def build_single_json(src = "output", dest= "michelin_images.json"):
    output_folder = src
    single_json = {
        'images': []
    }
    i = 0
    for name in os.listdir(output_folder):
        i += 1
        if i % 100 == 0:
            print(i)
        original_name = name[:-4] + "jpg"
        with open(os.path.join(output_folder, name), 'r') as jfile:
            # jdata = jfile.read()
            features = json.load(jfile)
            image = {
                'name': original_name,
                'features': features
            }
            single_json['images'].append(image)
    # pp(single_json)
    with open(dest, 'w') as f:
        json.dump(single_json, f)



def get_recursively(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found

def find_recursively(obj, query):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a value of the query
    provided.
    """
    # name = image['name']
    # features = image['features']
    fields_found = []

    for key, value in obj.items():

        if value == query:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = find_recursively(value, query)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = find_recursively(item, query)
                    for another_result in more_results:
                        fields_found.append(another_result)

    return fields_found

def find_label(labels, query):
    for label in labels:
        if label["description"] == query:
            return label

def find_object(objects, query):
    for obj in objects:
        if obj["name"] == query:
            return obj



def search(src="michelin_images.json", query="Plate", dest="index.json"):

    with open(src, 'r') as f:
        jdata = json.load(f)
    relevant_images = []
    scores = []
    for image in jdata['images']:
        # result = find_recursively(image, query)
        labels = image["features"]["labelAnnotations"]
        objects = image["features"]["localizedObjectAnnotations"]
        result_label = find_label(labels, query)
        result_obj = find_object(objects, query)
        if result_label or result_obj:
            name = image['name']
            relevant_images.append(name)
            label_score = result_label['score'] if result_label else 0
            obj_score = result_obj['score'] if result_obj else 0

            image_index = {
                'name': name,
                'labelScore': label_score,
                'objectScore': obj_score
            }
            scores.append(image_index)

    index = {
        "query": query,
        "images": relevant_images,
        "scores": scores
    }
    with open(dest, 'r') as f:
        index_data = json.load(f)

    index_data['queries'].append(index)

    with open(dest, 'w') as f:
        json.dump(index_data, f)



def review(index_path="index.json", query="Plate", data_path="data_all", resize=None):
    with open(index_path, 'r') as f:
        jdata = json.load(f)

    for index in jdata["queries"]:
        if index["query"] == query:
            break
    else:
        print("{} not found.".format(query))
        return
    relevant_images = index["images"]
    for image in relevant_images:
        path = os.path.join(data_path, image)
        image_rgb = imread(path)
        if resize:
            image_rgb = rescale(image_rgb, 0.1)
        plt.imshow(image_rgb)
        plt.show()






def main():
    im_path = "/Users/shimonheimowitz/PycharmProjects/michelin_star/data/54447129_2365265367027246_3685994048045175613_n.jpg"
    # detect_properties(im_path)
    # annanotate_image('data')
    # sample_async_batch_annotate_images("gs://michelin-data/data", "gs://michelin-features/data1")
    # annanotate_image_local('data_negative/foodphotography', "output/neg_med", lim=2000)
    # annanotate_image_local('data_lite', "output/neg")
    js = "foodphotography_images.json"
    build_single_json(src="output/neg_med", dest=js)
    query = "Plate"
    search(src=js, query=query, dest="index_neg_med.json")
    # review(query=query, resize=(100,100,3))
    query = "Dish"
    search(src=js, query=query, dest="index_neg_med.json")
    # review(query=query, resize=(100,100,3))

main()
