from bs4 import BeautifulSoup


def extract_text_from_html(html_content):
    """
    Extracts text content from given HTML code using the `BeautifulSoup` package.

    Parameters
    ----------
    html_content : str
        HTML code to parse
    """
    soup = BeautifulSoup(html_content, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join([p.get_text() for p in paragraphs])
    return text


def preprocess_google_nq(row):
    """
    Extracts text from HTML content for the Google NQ dataset.

    Parameters:
    ----------
    row : dict
        A row from the Google NQ dataset containing HTML content.

    Returns:
    -------
    str
        The extracted text content.
    """
    html_content = row["document"]["html"]
    return extract_text_from_html(html_content)


def get_google_nq_query(row):
    """
    Extracts the query from a row in the Google NQ dataset.

    Parameters:
    ----------
    row : dict
        A row from the Google NQ dataset.

    Returns:
    -------
    str
        The query.
    """
    return row["question"]["text"]


def get_google_nq_responses(row):
    """
    Extracts the expected responses from a row in the Google NQ dataset.

    Parameters:
    ----------
    row : dict
        A row from the Google NQ dataset.

    Returns:
    -------
    List[str]
        A list of expected responses.
    """
    short_answers = row["annotations"]["short_answers"]
    return [answer["text"][0] for answer in short_answers if answer["text"]]
