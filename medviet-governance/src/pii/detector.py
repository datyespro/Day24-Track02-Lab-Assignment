# src/pii/detector.py
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider


def build_vietnamese_analyzer() -> AnalyzerEngine:
    """
    Xay dung AnalyzerEngine voi cac recognizer tuy chinh cho VN.
    """

    # --- TASK 2.2.1 ---
    # Tao CCCD recognizer: so CCCD VN co 11 hoac 12 chu so
    # (pandas doc len 11 voi so co chu so 0 o dau)
    cccd_pattern = Pattern(
        name="cccd_pattern",
        regex=r"\d{11,12}",
        score=0.9
    )
    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[cccd_pattern],
        context=["cccd", "can cuoc", "chung minh", "cmnd"],
        supported_language="vi"
    )

    # --- TASK 2.2.2 ---
    # Tao phone recognizer: so dien thoai VN
    # Faker tao 9 chu so (khong co 0 dau), can catch 9-10 chu so
    phone_pattern = Pattern(
        name="vn_phone",
        regex=r"\d{9,10}",
        score=0.85
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[phone_pattern],
        context=["dien thoai", "sdt", "phone", "lien he"],
        supported_language="vi"
    )

    # --- Email recognizer cho tieng Viet ---
    # Built-in EmailRecognizer chi ho tro "en", nen tao custom cho "vi"
    email_pattern = Pattern(
        name="email_pattern",
        regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        score=0.85
    )
    email_recognizer = PatternRecognizer(
        supported_entity="EMAIL_ADDRESS",
        patterns=[email_pattern],
        context=["email", "e-mail", "thu dien tu"],
        supported_language="vi"
    )

    # --- Custom Vietnamese Name recognizer ---
    # xx_sent_ud_sm la sentence tokenizer, khong co NER.
    # Detect: 2-4 words starting with capital letter (Latin + Vietnamese diacritics).
    vn_name_pattern = Pattern(
        name="vn_name_pattern",
        regex=r"[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,3}",
        score=0.75
    )
    vn_name_recognizer = PatternRecognizer(
        supported_entity="VN_NAME",
        patterns=[vn_name_pattern],
        context=["ten", "ho ten", "benh nhan", "bac si"],
        supported_language="vi"
    )

    # --- TASK 2.2.3 ---
    # Tao NLP engine dung spaCy model (sentence tokenization)
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "vi",
                    "model_name": "xx_sent_ud_sm"}]
    })
    nlp_engine = provider.create_engine()

    # --- TASK 2.2.4 ---
    # Khoi tao AnalyzerEngine va add cac recognizer
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    analyzer.registry.add_recognizer(cccd_recognizer)
    analyzer.registry.add_recognizer(phone_recognizer)
    analyzer.registry.add_recognizer(email_recognizer)
    analyzer.registry.add_recognizer(vn_name_recognizer)

    return analyzer


def detect_pii(text: str, analyzer: AnalyzerEngine) -> list:
    """
    Detect PII trong text tieng Viet.
    Tra ve list cac RecognizerResult.
    Entities: PERSON (built-in), EMAIL_ADDRESS, VN_CCCD, VN_PHONE, VN_NAME
    """
    results = analyzer.analyze(
        text=text,
        language="vi",
        entities=["PERSON", "EMAIL_ADDRESS", "VN_CCCD", "VN_PHONE", "VN_NAME"]
    )
    return results
