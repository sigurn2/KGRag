
# 给出一段文本， 根据段落进行分段
def chunking_by_passage(
    content: str
) -> list[str]:
    return content.split('\n\n')