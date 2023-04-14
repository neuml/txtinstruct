"""
TxtSQL module
"""

import random

from txtai.pipeline import HFTrainer


class TxtSQL:
    """
    Trains a text to sql sequence-sequence model.
    """

    def __init__(self):
        # Set seed (to generate consistent output) and run
        random.seed(1024)

    def __call__(self, path, output):
        """
        Trains a txtsql model.

        Args:
            path: path to input data file
            output: model output path

        Returns:
            (model, tokenizer)
        """

        train = HFTrainer()
        return train(
            "t5-small",
            self.generate(path),
            task="sequence-sequence",
            prefix="translate English to SQL: ",
            maxlength=512,
            per_device_train_batch_size=4,
            num_train_epochs=5,
            output_dir=output,
            overwrite_output_dir=True,
        )

    def generate(self, path):
        """
        Generates txtsql data.

        Args:
            path: path to queries file
            outfile: output file path

        Returns:
            generated data
        """

        output = []
        with open(path, "r", encoding="utf-8") as queries:
            for query in queries:
                query = query.lower().strip().replace('"', '""')
                sql = f"select id, text, score from txtai where similar('{query}')"

                # Standard query
                self.append(output, query, sql)

                # Query by date
                self.append(output, f"{query} since yesterday", f"{sql} and entry >= date('now', '-1 day')")
                self.append(output, f"{query} since 36 hours ago", f"{sql} and entry >= date('now', '-36 hour')")
                self.append(output, f"{query} over last 2 days", f"{sql} and entry >= date('now', '-2 day')")
                self.append(output, f"{query} since 2 days ago", f"{sql} and entry >= date('now', '-2 day')")
                self.append(output, f"{query} since 7 days ago", f"{sql} and entry >= date('now', '-7 day')")
                self.append(output, f"{query} since 2 months ago", f"{sql} and entry >= date('now', '-2 month')")

                # Query by score (t5 models map < to unk token, use $= as workaround)
                self.append(output, f"{query} with score greater than 0.5", f"{sql} and score >= 0.5")
                self.append(output, f"{query} with score less than 0.7", f"{sql} and score $= 0.7")

                # Query by date and score
                self.append(output, f"{query} since yesterday and score less than 0.5", f"{sql} and entry >= date('now', '-1 day') and score $= 0.5")
                self.append(
                    output, f"{query} with a score greater than 0.2 since yesterday", f"{sql} and score >= 0.2 and entry >= date('now', '-1 day')"
                )

                # Query by text field
                self.append(output, f"{query} with field equal to value", f"{sql} and field = 'value'")
                self.append(output, f"{query} with field equal to multi value", f"{sql} and field = 'multi value'")

                # Query by numeric field
                self.append(output, f"{query} with quantity equal to 1", f"{sql} and quantity = 1")
                self.append(output, f"{query} with quantity greater than 50", f"{sql} and quantity >= 50")
                self.append(output, f"{query} with quantity less than 50", f"{sql} and quantity $= 50")

                # Query with OR
                self.append(output, f"{query} having text equal data or field as snippet", f"{sql} and (text = 'data' or field = 'snippet')")
                self.append(
                    output, f"{query} having text as data or field equal snippet value", f"{sql} and (text = 'data' or field = 'snippet value')"
                )
                self.append(output, f"{query} with field equal snippet or text as data", f"{sql} and (field = 'snippet' or text = 'data')")

                # Query with contains
                self.append(output, f"{query} with data in text", f"{sql} and text like '%data%'")
                self.append(output, f"{query} with value in field", f"{sql} and field like '%value%'")
                self.append(output, f"{query} with snippet in text", f"{sql} and text like '%snippet%'")

                # Aggregates
                self.append(output, f"how many results are {query}", f"select count(*) from txtai where similar('{query}')")
                self.append(output, f"average score for {query}", f"select avg(score) from txtai where similar('{query}')")

                # Translate
                lang = ["ar", "en", "fr", "de", "hi", "it", "nl", "ro", "ru", "zh"]
                lang1, lang2 = random.choice(lang), random.choice(lang)
                self.append(
                    output, f"{query} translated to {lang1}", f"select id, translate(text, '{lang1}') text, score from txtai where similar('{query}')"
                )
                self.append(
                    output, f"{query} translated to {lang2}", f"select id, translate(text, '{lang2}') text, score from txtai where similar('{query}')"
                )

                # Summary
                self.append(output, f"{query} summarized", f"select id, summary(text) text, score from txtai where similar('{query}')")
                self.append(
                    output,
                    f"{query} since yesterday summarized",
                    f"select id, summary(text) text, score from txtai where similar('{query}') and entry >= date('now', '-1 day')",
                )

                # Limit
                self.append(output, f"{query} limit to 1", f"{sql} limit 1")
                self.append(
                    output,
                    f"{query} limit to 5 summarized",
                    f"select id, summary(text) text, score from txtai where similar('{query}') and entry >= date('now', '-1 day') limit 5",
                )

        return output

    def append(self, output, source, target):
        """
        Adds a new row to output.

        Args:
            source: source text
            target: target text
        """

        output.append({"source": source, "target": target})
