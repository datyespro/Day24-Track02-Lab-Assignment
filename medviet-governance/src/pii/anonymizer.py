import hashlib
import re

import pandas as pd
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from faker import Faker

from .detector import build_vietnamese_analyzer, detect_pii

fake = Faker("vi_VN")


class MedVietAnonymizer:

    def __init__(self):
        self.analyzer = build_vietnamese_analyzer()
        self.anonymizer = AnonymizerEngine()

    def _mask_text(self, text: str, start: int, end: int) -> str:
        """Mask characters between start and end with asterisks."""
        return text[:start] + "*" * (end - start) + text[end:]

    def _mask_entity_in_text(self, text: str, entity_type: str, start: int, end: int) -> str:
        """Apply masking strategy to a detected entity span."""
        entity_text = text[start:end]
        if entity_type == "PERSON":
            words = entity_text.split()
            masked = []
            for word in words:
                if len(word) <= 1:
                    masked.append(word)
                else:
                    masked.append(word[0] + "*" * (len(word) - 1))
            return " ".join(masked)
        elif entity_type in ("VN_CCCD", "VN_PHONE"):
            return "*" * len(entity_text)
        elif entity_type == "EMAIL_ADDRESS":
            at = entity_text.find("@")
            if at > 0:
                local = entity_text[:at]
                domain = entity_text[at:]
                masked_local = local[0] + "*" * (len(local) - 1) if len(local) > 1 else local
                return masked_local + domain
            return "*" * len(entity_text)
        return "*" * (end - start)

    def anonymize_text(self, text: str, strategy: str = "replace") -> str:
        """
        Anonymize text with the chosen strategy.

        Strategies:
        - "mask"    : Nguyen Van A → N****** V** A
        - "replace" : thay bang fake data (dung Faker)
        - "hash"    : SHA-256 one-way hash
        - "generalize": chi dung cho tuoi/nam sinh
        """
        results = detect_pii(text, self.analyzer)
        if not results:
            return text

        if strategy == "replace":
            operators = {
                "PERSON": OperatorConfig("replace",
                          {"new_value": fake.name()}),
                "EMAIL_ADDRESS": OperatorConfig("replace",
                                 {"new_value": fake.email()}),
                "VN_CCCD": OperatorConfig("replace",
                           {"new_value": fake.ssn()}),
                "VN_PHONE": OperatorConfig("replace",
                            {"new_value": fake.phone_number()}),
                "VN_NAME": OperatorConfig("replace",
                          {"new_value": fake.name()}),
            }
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )
            return anonymized.text

        elif strategy == "mask":
            masked_text = text
            for result in sorted(results, key=lambda r: r.start, reverse=True):
                entity_type = result.entity_type
                entity_text = masked_text[result.start:result.end]
                if entity_type in ("PERSON", "VN_NAME"):
                    words = entity_text.split()
                    masked_words = []
                    for word in words:
                        if len(word) <= 1:
                            masked_words.append(word)
                        else:
                            masked_words.append(word[0] + "*" * (len(word) - 1))
                    replacement = " ".join(masked_words)
                elif entity_type in ("VN_CCCD", "VN_PHONE"):
                    replacement = "*" * len(entity_text)
                elif entity_type == "EMAIL_ADDRESS":
                    at = entity_text.find("@")
                    if at > 0:
                        local = entity_text[:at]
                        domain = entity_text[at:]
                        ml = local[0] + "*" * (len(local) - 1) if len(local) > 1 else local
                        replacement = ml + domain
                    else:
                        replacement = "*" * len(entity_text)
                else:
                    replacement = "*" * len(entity_text)
                masked_text = masked_text[:result.start] + replacement + masked_text[result.end:]
            return masked_text

        elif strategy == "hash":
            operators = {
                "PERSON": OperatorConfig("hash", {"hash_type": "sha256"}),
                "EMAIL_ADDRESS": OperatorConfig("hash", {"hash_type": "sha256"}),
                "VN_CCCD": OperatorConfig("hash", {"hash_type": "sha256"}),
                "VN_PHONE": OperatorConfig("hash", {"hash_type": "sha256"}),
                "VN_NAME": OperatorConfig("hash", {"hash_type": "sha256"}),
            }
            anonymized = self.anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators
            )
            return anonymized.text

        return text

    def anonymize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize toan bo DataFrame.
        - Cot text (ho_ten, dia_chi, email): dung anonymize_text()
        - Cot cccd, so_dien_thoai: replace truc tiep bang fake data
        - Cot benh, ket_qua_xet_nghiem: GIU NGUYEN (can cho model training)
        - Cot patient_id: GIU NGUYEN (pseudonym da du an toan)
        """
        df_anon = df.copy()

        df_anon["ho_ten"] = df_anon["ho_ten"].apply(
            lambda x: self.anonymize_text(str(x), strategy="replace")
        )
        df_anon["dia_chi"] = df_anon["dia_chi"].apply(
            lambda x: self.anonymize_text(str(x), strategy="replace")
        )
        df_anon["email"] = df_anon["email"].apply(
            lambda x: self.anonymize_text(str(x), strategy="replace")
        )
        df_anon["cccd"] = [fake.ssn() for _ in range(len(df_anon))]
        df_anon["so_dien_thoai"] = [fake.phone_number() for _ in range(len(df_anon))]

        return df_anon

    def calculate_detection_rate(self,
                                  original_df: pd.DataFrame,
                                  pii_columns: list) -> float:
        """
        Tinh % PII duoc detect thanh cong.
        Muc tieu: > 95%

        Logic: voi moi o trong pii_columns,
               kiem tra xem detect_pii() co tim thay it nhat 1 entity khong.
        """
        total = 0
        detected = 0

        for col in pii_columns:
            for value in original_df[col].astype(str):
                total += 1
                results = detect_pii(value, self.analyzer)
                if len(results) > 0:
                    detected += 1

        return detected / total if total > 0 else 0.0
