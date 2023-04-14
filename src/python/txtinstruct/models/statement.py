"""
Statement module
"""

from string import Formatter

from datasets import Dataset

from txtai.pipeline import HFTrainer


class StatementGenerator:
    """
    Trains a statement generator model.
    """

    def __call__(self, base, data, task, prompt=None, **kwargs):
        """
        Train a statement generator model.

        Args:
            base: input model or model path
            data: instruction-tuning dataset
            task: model task
            prompt: optional prompt template, uses default when not provided
            kwargs: additional training arguments, see HFTrainer docs

        Returns:
            (model, tokenizer)
        """

        # Get prompt
        prompt = prompt if prompt else self.defaultprompt(task)

        # Build training dataset
        train = Dataset.from_generator(self.generate, gen_kwargs=({
            "data": data,
            "task": task,
            "prompt": prompt
            })
        )

        # Train model
        trainer = HFTrainer()
        return trainer(base, train, task=task, **kwargs)

    def generate(self, data, task, prompt):
        """
        Generates a statement generation dataset for training. This method generates fields based on the model task.

        Args:
            data: statement generation dataset
            task: model task
            prompt: input prompt template
        """

        # Template formatter
        formatter = Formatter()

        for row in data:
            # Generate question context
            context = row["context"]

            if task == "language-generation":
                yield {
                    "text": formatter.format(prompt, context=context) + row["question"]
                }
            else:
                yield {
                    "source": formatter.format(prompt, context=context),
                    "target": row["question"]
                }

    def defaultprompt(self, task):
        prompt = """Generate a question using the context below.
    ### Context:
    {context}
    """

        if task == "language-generation":
            prompt += "\n### Answer:\n"

        return prompt
