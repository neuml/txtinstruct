"""
BashSQL module
"""

import random

from txtai.pipeline import HFTrainer


class BashSQL:
    """
    Trains a bash to sql sequence to sequence model.
    """

    def __init__(self):
        """
        Creates a new BashSQL instance.
        """

        # Set seed (to generate consistent output) and run
        random.seed(1024)

    def __call__(self, path, output):
        """
        Trains a bashsql model.

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
            prefix="translate Bash to SQL: ",
            maxlength=512,
            per_device_train_batch_size=4,
            num_train_epochs=5,
            output_dir=output,
            overwrite_output_dir=True,
        )

    def generate(self, path):
        """
        Generates bashsql data.

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
                find = f'find -name ""{query}""' if " " in query else f"find -name {query}"
                sql = f"select id, text, score from txtai where similar('{query}')"

                # Standard query
                self.append(output, find, sql)

                # Query by date
                self.append(output, f"{find} -mtime -1", f"{sql} and entry >= date('now', '-1 day')")
                self.append(output, f"{find} -mtime -1.5", f"{sql} and entry >= date('now', '-1.5 day')")
                self.append(output, f"{find} -mtime -2", f"{sql} and entry >= date('now', '-2 day')")

                # Query by score (t5 models map < to unk token, use $= as workaround)
                self.append(output, f"{find} -score +0.5", f"{sql} and score >= 0.5")
                self.append(output, f"{find} -score -0.7", f"{sql} and score $= 0.7")

                # Query by date and score
                self.append(output, f"{find} -mtime -1 -score -0.5", f"{sql} and entry >= date('now', '-1 day') and score $= 0.5")
                self.append(output, f"{find} -score +0.2 -mtime -1", f"{sql} and score >= 0.2 and entry >= date('now', '-1 day')")

                # Query by text field
                self.append(output, f"{find} -field value", f"{sql} and field = 'value'")
                self.append(output, f'{find} -field "multi value"', f"{sql} and field = 'multi value'")

                # Query by numeric field
                self.append(output, f"{find} -quantity 1", f"{sql} and quantity = 1")
                self.append(output, f"{find} -quantity +50", f"{sql} and quantity >= 50")
                self.append(output, f"{find} -quantity -50", f"{sql} and quantity $= 50")

                # Query with contains
                self.append(output, f"{find} -text ~data", f"{sql} and text like '%data%'")
                self.append(output, f"{find} -field ~value", f"{sql} and field like '%value%'")
                self.append(output, f"{find} -text ~snippet", f"{sql} and text like '%snippet%'")

                # Aggregates
                self.append(output, f"{find} -count", f"select count(*) from txtai where similar('{query}')")
                self.append(output, f"{find} -average", f"select avg(score) from txtai where similar('{query}')")

                # Translate
                lang = ["ar", "en", "fr", "de", "hi", "it", "nl", "ro", "ru", "zh"]
                lang1, lang2 = random.choice(lang), random.choice(lang)
                self.append(
                    output, f"{find} -translate {lang1}", f"select id, translate(text, '{lang1}') text, score from txtai where similar('{query}')"
                )
                self.append(
                    output, f"{find} -translate {lang2}", f"select id, translate(text, '{lang2}') text, score from txtai where similar('{query}')"
                )
                self.append(
                    output,
                    f"{find} -mtime -1 -field 0 -translate {lang1}",
                    f"select id, translate(text, '{lang1}') text, score from txtai where similar('{query}') "
                    + "and entry >= ('now', '-1 day') and field = 0",
                )

                # Summary
                self.append(output, f"{find} -summary", f"select id, summary(text) text, score from txtai where similar('{query}')")
                self.append(
                    output,
                    f"{find} -mtime -1 -summary",
                    f"select id, summary(text) text, score from txtai where similar('{query}') and entry >= date('now', '-1 day')",
                )

                # Limit
                self.append(output, f"{find} -limit 1", f"{sql} limit 1")
                self.append(
                    output,
                    f"{find} -limit 5 -summary",
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
