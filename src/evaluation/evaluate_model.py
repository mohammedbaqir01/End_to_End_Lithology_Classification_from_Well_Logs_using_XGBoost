from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pylab as plt


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a trained model on test data.

    Parameters:
    - model: The trained model to evaluate.
    - X_test: Test features.
    - y_test: True labels for the test data.

    Returns:
    - report: A dictionary containing the classification report.
    """

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    return report 
