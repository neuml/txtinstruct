"""
Data module
"""

import json
import random

from string import Formatter

from tqdm import tqdm


class DatasetBuilder:
    """
    Generates an instruction dataset using statement generation and text generation models.
    """

    def __init__(self, model, statement, templates=None, prompt=None, sprompt=None):
        """
        Creates a new DatasetBuilder.

        Args:
            model: target text generation model
            statement: statement generation model
            templates: optional list of custom statement templates
            prompt: optional model prompt
            sprompt: optional custom statement prompt
        """

        # Target text generation model
        self.model = model

        # Statement generation model
        self.statement = statement

        # Target text prompt
        self.prompt = prompt if prompt else self.defaultprompt(self.model)

        # Statement prompt
        self.sprompt = sprompt if sprompt else self.defaultsprompt(self.statement)

        # Statement templates
        self.templates = templates

        # Template formatter
        self.formatter = Formatter()

        # Set random seed generated data is deterministic
        random.seed(42)

    def __call__(self, rows, total, output):
        """
        Build a dataset with input rows.

        Args:
            rows: iterable of {id, text}
            total: total number of rows expected
            output: output file path
        """

        batch, outputs = [], []

        for row in tqdm(rows, total=total):
            batch.append(row)

            # Generate content for batch
            if len(batch) == 64:
                outputs.extend(self.generate(batch))
                batch = []

        # Last batch
        if batch:
            outputs.extend(self.generate(batch))

        # Write file
        with open(output, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=4)

    def generate(self, rows):
        """
        Generates targets for a batch of input rows.

        Args:
            rows: batch of rows

        Returns:
            outputs
        """

        # Split into ids and texts
        ids = [row["id"] for row in rows]
        texts = [row["text"] for row in rows]

        # Generate statements
        statements = self.statement([
            self.formatter.format(self.sprompt, context=row["text"]) for row in rows],
            truncation=True,
            batch_size=len(rows)
        )

        # Generate template statements
        templates = self.template(ids) if self.templates else []

        # Create prompts
        queue, prompts = [], []
        for x, text in enumerate(texts):
            # Generate statement prompt
            prompts.append(self.formatter.format(self.prompt, statement=statements[x], context=text))

            # Generate template statement prompt
            if templates:
                prompts.append(self.formatter.format(self.prompt, statement=templates[x], context=text))

            # Store all generated statements as single row
            queue.append([statements[x], templates[x]] if templates else [statements[x]])

        # Generate target text from prompts
        targets = self.model(prompts, truncation=True, batch_size=8)

        # Answer index
        index, outputs = 0, []
        for x, text in enumerate(texts):
            output = {"context": text, "statements": []}
            for question in queue[x]:
                output["statements"].append({
                    "source": question,
                    "target": targets[index]
                })

                index += 1

            # Generate unanswerable statement
            y = random.choice([i for i in range(0, len(texts)) if i != x])
            statement = random.choice([statements[y], templates[y]]) if templates else statements[y]
            output["statements"].append({
                "source": statement,
                "target": "I don't have data on that"
            })

            outputs.append(output)

        return outputs

    def template(self, ids):
        """
        Generates template statements using ids as the input text. This method assumes each id is a text identifier.

        Args:
            ids: list of ids

        Returns:
            generated template statements
        """

        # Generate statements and run
        statements = []
        for uid in ids:
            # Get query template
            template = random.choice(self.templates)

            # Create statement
            statements.append(self.formatter.format(template, text=uid))

        return statements

    def defaultsprompt(self, model):
        """
        Default statement prompt when otherwise not provided

        Args:
            model: statement generation model

        Returns:
            default statement prompt
        """

        prompt = """Generate a question using the context below.
    ### Context:
    {context}
    """

        # Infer model task
        task = self.infertask(model)

        if task == "language-generation":
            prompt += "\n### Answer:\n"

        return prompt

    def defaultprompt(self, model):
        """
        Default target text generation prompt.

        Args:
            model: target text generation model

        Returns:
            default target text generation prompt
        """

        # Infer model task
        task = self.infertask(model)

        template = "Answer the following question using only the context below. Give a detailed answer. "
        template += "Say 'I don't have data on that' when the question can't be answered.\n"
        template += "Question: {statement}\n"
        template += "Context: {context}"
        template += "\nAnswer: " if task == "language-generation" else ""
        return template

    def infertask(self, model):
        """
        Infer the model task (language-generation or sequence-sequence) using model configuration.

        Args:
            model: input model

        Returns:
            model task
        """

        architecture = model.config.architectures[0] if model.config.architectures else None

        if any(x for x in ["LMHead", "CausalLM"] if x in architecture):
            return "language-generation"

        return "sequence-sequence"
