from pathlib import Path

from data import read_qa_data


CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data" / "hw7"


def main() -> None:
    expected_counts = {
        "hw7_train.json": 31690,
        "hw7_dev.json": 4131,
        "hw7_test.json": 4957,
    }
    for filename, expected_count in expected_counts.items():
        questions, paragraphs = read_qa_data(DATA_DIR / filename)
        print(
            f"{filename}: questions={len(questions)}, "
            f"paragraphs={len(paragraphs)}"
        )
        if len(questions) != expected_count:
            raise ValueError(
                f"{filename} should contain {expected_count} questions."
            )

    train_questions, train_paragraphs = read_qa_data(DATA_DIR / "hw7_train.json")
    for question in train_questions:
        paragraph = train_paragraphs[question["paragraph_id"]]
        extracted = paragraph[
            question["answer_start"] : question["answer_end"] + 1
        ]
        if extracted != question["answer_text"]:
            raise ValueError(
                f"Answer offset mismatch for train question {question['id']}: "
                f"{extracted!r} != {question['answer_text']!r}"
            )
    print("All training answer offsets are valid.")


if __name__ == "__main__":
    main()

