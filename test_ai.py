from app import process_uploaded_files


def test_ai():
    _, text = process_uploaded_files('t.jpg')
    assert text == 'liner, ocean liner'
