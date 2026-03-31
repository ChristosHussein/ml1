import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

def train_and_evaluate_nn(X_train, y_train, X_val, y_val):
    """
    Task 3.2: Trains a Neural Network with 2 hidden layers, ReLU, Dropout, and Early Stopping.
    """
    print("\n--- Task 3.2: Training Neural Network ---")

    # 1. Build the Architecture
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # 2. Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    # 3. Setup Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 4. Train the model
    print("Training Neural Network... (This will print epochs as it learns)")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )

    # 5. Plot the Training and Validation Loss Curves
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('nn_loss_curve.png')
    plt.close()
    print("\nSaved Loss Curve to 'nn_loss_curve.png'")
    

    # 6. Evaluate on Validation Set
    val_probs = model.predict(X_val).flatten()
    val_preds = (val_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_probs)

    print(f"NN Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"NN Validation ROC-AUC:  {roc_auc:.4f}")
    print("\nDetailed NN Classification Report:")
    print(classification_report(y_val, val_preds))
    
    # 7. Generate a Confusion Matrix for the NN
    cm = confusion_matrix(y_val, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Returned", "Returned"])
    disp.plot(cmap='Oranges')
    plt.title('Neural Network Confusion Matrix (Validation)')
    plt.savefig('nn_confusion_matrix.png')
    plt.close()
    print("Saved NN Confusion Matrix to 'nn_confusion_matrix.png'")

    # 8. Save the model in the required .h5 format
    model.save('models/neural_network.h5')
    print("Neural Network saved to 'models/neural_network.h5'")

    return model

def test_nn_model(model, X_test, y_test):
    """
    Task 4: Evaluates the neural network on the final test set.
    """
    print("\n--- Task 4: Final Test Set Evaluation (Neural Network) ---")
    
    test_probs = model.predict(X_test).flatten()
    test_preds = (test_probs > 0.5).astype(int)

    accuracy = accuracy_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_probs)

    print(f"Final NN Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Final NN Test ROC-AUC:  {roc_auc:.4f}")
    print("\nFinal NN Test Classification Report:")
    print(classification_report(y_test, test_preds))