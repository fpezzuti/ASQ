from dataclasses import dataclass

def extract_between_tags(text, start_tag, end_tag):
    import re
    pattern = re.compile(rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None

@dataclass(frozen=True)
class ControlTAGs:
    QUERY: str = "search"
    ANSWER: str = "answer"
    THINK: str = "think"
    RANKEDLIST: str = "information"

    OPEN_TAG_TEMPLATE: str = "<{tag_name}>"
    CLOSE_TAG_TEMPLATE: str = "</{tag_name}>"

    @property
    def tags(self) -> set[str]:
        return {
            self.QUERY,
            self.ANSWER,
            self.THINK,
            self.RANKEDLIST,
        }

    def _validate(self, tag: str) -> None:
        if tag not in self.tags:
            raise ValueError(f"Unknown control tag: {tag}")

    def get_open_tag(self, tag: str) -> str:
        self._validate(tag)
        return self.OPEN_TAG_TEMPLATE.format(tag_name=tag)

    def get_close_tag(self, tag: str) -> str:
        self._validate(tag)
        return self.CLOSE_TAG_TEMPLATE.format(tag_name=tag)

    def wrap(self, tag: str, content: str) -> str:
        return f"{self.get_open_tag(tag)}{content}{self.get_close_tag(tag)}"

    def extract(self, tag: str, text: str) -> str | None:
        return extract_between_tags(
            text,
            self.get_open_tag(tag),
            self.get_close_tag(tag),
        )

class RegexExtractor():
    
    def __init__(self, ctrl_tags: ControlTAGs):
        self.control_tags: ControlTAGs = ctrl_tags
    
    def get_query(self, text):
        return self.control_tags.extract(self.control_tags.QUERY, text)
    
    def get_answer(self, text):
        return self.control_tags.extract(self.control_tags.ANSWER, text)
    
    def get_thought(self, text):
        return self.control_tags.extract(self.control_tags.THINK, text)