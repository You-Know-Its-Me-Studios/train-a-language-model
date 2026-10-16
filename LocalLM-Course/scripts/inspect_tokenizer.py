from tokenizers import Tokenizer

from prepare_tokens import EOT, OUT, verify


def main():
    verify()
    tokenizer = Tokenizer.from_file(
        str(OUT / "tokenizer.json")
    )

    text = "The little dog found a red ball."
    encoded = tokenizer.encode(
        text, add_special_tokens=False
    )
    decoded = tokenizer.decode(encoded.ids)

    print("Original:", ascii(text))
    print("Token IDs:", encoded.ids)
    print("Token pieces:", ascii(encoded.tokens))
    print("Token count:", len(encoded.ids))
    print("Decoded:", ascii(decoded))

    if decoded != text:
        raise RuntimeError("Round trip failed.")
    print("Round trip passed.")

    eot_id = tokenizer.token_to_id(EOT)
    with_boundary = encoded.ids + [eot_id]
    visible = tokenizer.decode(
        with_boundary, skip_special_tokens=False
    )
    print("With end marker:", ascii(visible))


if __name__ == "__main__":
    main()
