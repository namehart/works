with open("API/doc.md","+r",encoding="utf-8") as f:
    __doc__ = f.read()
def print_doc():
    """
    打印文档内容。
    """
    print(__doc__)

if __name__ == "__main__":
    print_doc()