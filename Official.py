
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
import numpy as np
import csv
from keras import Sequential
from keras import layers
import webbrowser
import pandas as pd
import requests

# URLs for your files on Google Cloud Storage (replace these with actual public URLs)
careers_csv_url = "https://storage.googleapis.com/hack4eduprofrom/Careers.csv"
engineer_personality_dataset_csv_url = "https://storage.googleapis.com/hack4eduprofrom/engineer_personality_dataset.csv"
career_model_url = "https://storage.googleapis.com/hack4eduprofrom/career_model.keras"
career_engineer_model_url = "https://storage.googleapis.com/hack4eduprofrom/career_engineer_model.keras"

# Function to download and save a file from a given URL
def download_file(url, local_filename):
    print(f"Downloading {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded and saved as {local_filename}")
    else:
        print(f"Failed to download {url}")

# Download CSV and model files from Google Cloud Storage
download_file(careers_csv_url, "Careers.csv")
download_file(engineer_personality_dataset_csv_url, "engineer_personality_dataset.csv")
download_file(career_model_url, "career_model.keras")
download_file(career_engineer_model_url, "career_engineer_model.keras")

# Load the CSV files into Pandas DataFrames
careers_df = pd.read_csv("Careers.csv")
engineer_personality_df = pd.read_csv("engineer_personality_dataset.csv")

# Load the models
career_model = keras.models.load_model("career_model.keras")
career_engineer_model = keras.models.load_model("career_engineer_model.keras")

# Now you can proceed with the rest of your original logic as it was...


TEST_SIZE = 0.4
EPOCHS = 100
MODEL_PATH = "career_model.keras" # Path where the model will be saved/loaded
MODEL_ENGINEER_PATH = "career_engineer_model.keras"
def main():
    
    
    # Load evidence and labels from the dataset
    evidence, labels = load_data(careers_df)

    # Convert evidence and labels to NumPy arrays for model training
    evidence = np.array(evidence, dtype=float)
    labels = np.array(labels)

    # Encode the labels
    label_encoder = LabelEncoder()
    labels_encode = label_encoder.fit_transform(labels)
    labels_categorical = keras.utils.to_categorical(labels_encode)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        evidence, labels_categorical, test_size=TEST_SIZE
    )
    
    # Get and compile the model
    model = get_model(x_train.shape[1], len(label_encoder.classes_))
    
    # Train the model
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(x_test, y_test))

    # Save the model
    model.save(MODEL_PATH)
    
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

    # Provide the option to make predictions with user input
    while True:
        user_choice = input("Do you want to make a prediction? (yes/no): ").strip().lower()
        if user_choice == "yes":
            prediction = make_prediction(label_encoder)
            
            if prediction == "Engineer":
                evidence2, labels2 = load_data2(engineer_personality_df)
                evidence2 = np.array(evidence2, dtype=float)
                labels2 = np.array(labels2) 
                
                #Encode Labels
                label_encoder2 = LabelEncoder()
                labels_encode2 = label_encoder2.fit_transform(labels2)
                labels_categorical2 = keras.utils.to_categorical(labels_encode2)
                
                #Split data
                x2_train, x2_test, y2_train, y2_test = train_test_split(
                    evidence2, labels_categorical2, test_size=TEST_SIZE
                )          
                    
                Engineer_model = get_modelEngineer(x2_train.shape[1], len(label_encoder2.classes_))
                
                Engineer_model.fit(x2_train,y2_train,epochs=EPOCHS, batch_size=32, validation_data=(x2_test,y_test))
                
                Engineer_model.save(MODEL_ENGINEER_PATH)
                
                test_loss2, test_acc2 = Engineer_model.evaluate(x2_test, y2_test)
                print(f"Test accuracy: {test_acc2}")
                print(f"Test loss: {test_loss2}")
                
                
                prediction2 = make_engineer_prediction(label_encoder2)
                
                if prediction2 == 'Software Engineer':
                    choice = int(input("What do you want to learn: BackEnd(1), FrontEnd(2), Applications(3), AI(4)"))
                    
                    if choice == 1:
                        webbrowser.open(
                            "https://www.edx.org/learn/computer-science/harvard-university-cs50-s-introduction-to-computer-science?linked_from=autocomplete-prequery&c=autocomplete-prequery&position=1"
                        )
                    elif choice == 2:
                        
                        webbrowser.open(
                            "https://www.edx.org/es/certificates/professional-certificate/w3cx-front-end-web-developer?index=spanish_product&queryID=51887fe81a69cf7085b121107ccff344&position=2&linked_from=autocomplete&c=autocomplete"
                        )
                    elif choice == 3:
                        webbrowser.open(
                            "https://www.edx.org/es/learn/django/ibm-django-application-development-with-sql-and-databases?index=spanish_product&queryID=254642d32b3012a4e3f4685d43159f33&position=7&linked_from=autocomplete&c=autocomplete"
                        )
                    elif choice == 4:
                        
                        webbrowser.open(
                            "https://www.edx.org/es/learn/artificial-intelligence/harvard-university-cs50-s-introduction-to-artificial-intelligence-with-python?index=spanish_product&queryID=15bb017ce76a6d95f4dbec751c3dfab2&position=3&results_level=first-level-results&term=cs50&objectID=course-3a31db71-de8f-45f1-ae65-11981ed9d680&campaign=CS50%27s+Introduction+to+Artificial+Intelligence+with+Python&source=edX&product_category=course&placement_url=https%3A%2F%2Fwww.edx.org%2Fes%2Fsearch"
                        )
                        webbrowser.open(
                            "https://www.deeplearning.ai/courses/data-engineering/"
                        )
                          
            
        else:
            print("Exiting...")
            break
    


def load_data(df):
    # Use the relevant columns for evidence and labels
    evidence = df[['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']]
    labels = df['Best Career']
    
    return evidence, labels

def load_data2(df):
    # Use the relevant columns for evidence and labels in the engineer personality dataset
    evidence = df[['Manipulation Capacity', 'Analitical Capacity', 'Computer Skills', 'Teamwork', 'Self Learning']]
    labels = df['Best Career']
    
    return evidence, labels


    
    # Open and read the CSV file
    
    
def get_model(input_size, num_classes):
    # Define the model using the Sequential API
    model = Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_size,)))
    
    # Hidden layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def make_prediction(label_encoder):
    # Load the trained model
    model = keras.models.load_model(MODEL_PATH)
    
    # Ask the user to input the values for the 6 features
    user_input = []
    traits = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    for trait in traits:
        value = float(input(f"Enter your score for {trait} (0-10): ").strip())
        user_input.append(value)
    
    # Convert the input to the right shape for the model
    user_input = np.array([user_input])  # 2D array

    # Make a prediction
    prediction = model.predict(user_input)
    predicted_class = np.argmax(prediction)  # Get the class index
    
    # Decode the predicted class to the original label
    career_prediction = label_encoder.inverse_transform([predicted_class])[0]
    
    # Output the prediction
    print(f"The best career prediction based on your input is: {career_prediction}")
    
    return career_prediction


def get_modelEngineer(input_size, numclass ):
    model = Sequential()
    #input
    model.add(layers.Input(shape=(input_size,)))
    
    #Hidden Layers
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    
    #output
    model.add(layers.Dense(numclass, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
    
def make_engineer_prediction(label_encoder2):
    model = keras.models.load_model(MODEL_ENGINEER_PATH)
    
    user_input = []
    traits = ['Manipulation Capacity', 'Analitical Capacity', 'Computer Skills', 'Teamwork', 'Self Learning']
    for trait in traits:
        value = float(input(f'Enter your score in this {trait} (0-10): ').strip())
        user_input.append(value)
    
    user_input = np.array([user_input])
    prediction = model.predict(user_input)
    predicted_class = np.argmax(prediction)
    
    career_prediction = label_encoder2.inverse_transform([predicted_class])[0]
    
    print(f'Career {career_prediction}')
    return career_prediction
if __name__ == "__main__":
    main()
