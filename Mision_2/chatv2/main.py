from chatbot.data import training_data
from chatbot.model import build_and_train_model, load_model, predict_answer


def main():
    # Cargar modelo guardado
    model, vectorizer, unique_answers = load_model()

    # Si no existe, entrenar modelo nuevo
    if model is None:
        model, vectorizer, unique_answers = build_and_train_model(training_data)

    print("\nChatbot listo. Escribe 'salir' para terminar.\n")

    while True:
        user = input("Tú: ").strip()

        if user.lower() in ("salir", "exit", "quit"):
            print("Bot: Hasta pronto")
            break

        response = predict_answer(model, vectorizer, unique_answers, user)
        print("Bot:", response)


if __name__ == "__main__":
    main()