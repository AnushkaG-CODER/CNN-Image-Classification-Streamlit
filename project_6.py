
# =====================================================
# PART A: CNN IMAGE CLASSIFICATION USING MNIST DATASET
# =====================================================

def run_cnn():

    import tensorflow as tf
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt
    import numpy as np

    # Step 1: Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    print("Number of training images:", len(X_train))
    print("Number of testing images:", len(X_test))

    print("Classes:", np.unique(y_train))

    # Step 2: Display sample images
    plt.figure(figsize=(10, 4))

    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_train[i], cmap="gray")
        plt.title("Label: " + str(y_train[i]))
        plt.axis("off")

    plt.show(block=False)
    plt.pause(3)
    plt.close()

    print("Sample images displayed successfully.")

    # Step 3: Preprocess images
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    # Step 4: Create CNN model
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    # Step 5: Display model summary
    model.summary()

    # Step 6: Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Step 7: Train model
    history = model.fit(
        X_train,
        y_train,
        epochs=3,
        validation_data=(X_test, y_test)
    )

    # Step 8: Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # Step 9: Plot accuracy graph
    plt.figure(figsize=(6, 4))

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show(block=False)
    plt.pause(5)
    plt.close()

    print("Accuracy graph completed.")

    # Step 10: Plot loss graph
    plt.figure(figsize=(6, 4))

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Step 11: Predict test images
    predictions = model.predict(X_test)

    predicted_labels = np.argmax(predictions, axis=1)

    # Step 12: Find correct and incorrect predictions
    correct = np.where(predicted_labels == y_test)[0]
    incorrect = np.where(predicted_labels != y_test)[0]

    print("Correct Predictions:", len(correct))
    print("Incorrect Predictions:", len(incorrect))

    # Step 13: Display correct predictions
    print("Correct Predictions")

    plt.figure(figsize=(10, 4))

    for i in range(5):
        index = correct[i]

        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[index].reshape(28, 28), cmap="gray")

        plt.title(
            "Actual: " + str(y_test[index]) +
            "\nPredicted: " + str(predicted_labels[index])
        )

        plt.axis("off")

    plt.show()

    # Step 14: Display incorrect predictions
    print("Incorrect Predictions")

    if len(incorrect) > 0:

        plt.figure(figsize=(10, 4))

        for i in range(min(5, len(incorrect))):
            index = incorrect[i]

            plt.subplot(1, 5, i + 1)
            plt.imshow(X_test[index].reshape(28, 28), cmap="gray")

            plt.title(
                "Actual: " + str(y_test[index]) +
                "\nPredicted: " + str(predicted_labels[index])
            )

            plt.axis("off")

        plt.show()

    print("CNN Classification Completed!")


# =====================================================
# PART B: STREAMLIT STUDENT MARKS CALCULATOR
# =====================================================

def run_streamlit():

    import streamlit as st

    st.set_page_config(page_title="Student Marks Calculator")

    st.title("Student Marks Calculator")

    st.write("Enter student details and subject marks.")

    # Student name
    name = st.text_input("Enter Student Name")

    # Subject marks
    st.subheader("Enter Subject Marks")

    m1 = st.number_input("Subject 1", min_value=0, max_value=100)
    m2 = st.number_input("Subject 2", min_value=0, max_value=100)
    m3 = st.number_input("Subject 3", min_value=0, max_value=100)
    m4 = st.number_input("Subject 4", min_value=0, max_value=100)
    m5 = st.number_input("Subject 5", min_value=0, max_value=100)

    # Calculate button
    if st.button("Calculate Result"):

        if name.strip() == "":
            st.warning("Please enter student name.")

        else:
            total = m1 + m2 + m3 + m4 + m5

            percentage = total / 5

            st.subheader("Student Result")

            st.write("Student Name:", name)
            st.write("Total Marks:", total, "/ 500")
            st.write("Percentage:", percentage, "%")

            if min(m1, m2, m3, m4, m5) >= 35:
                st.success("Result: PASS")
            else:
                st.error("Result: FAIL")


# =====================================================
# MAIN PROGRAM
# =====================================================

if __name__ == "__main__":

    import sys

    if "--cnn" in sys.argv:
        run_cnn()

    else:
        run_streamlit()
