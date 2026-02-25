import polars as pl
from typing import Dict, List

from pyhealth.tasks.base_task import BaseTask


class CorGANGenerationMIMIC3(BaseTask):
    """Task function for CorGAN synthetic EHR generation using MIMIC-III.

    Extracts ICD-9 diagnosis codes from MIMIC-III admission records into a
    nested visit structure suitable for training the CorGAN model.

    Each sample contains the full visit history for a single patient, where
    each visit is a list of ICD-9 codes recorded during that admission.
    Patients with fewer than 2 visits are excluded.

    Attributes:
        task_name (str): Unique task identifier.
        input_schema (dict): Schema descriptor for the visits field.
        output_schema (dict): Empty — generative task, no conditioning label.
        _icd_col (str): Polars column path for ICD codes in MIMIC-III.

    Examples:
        >>> fn = CorGANGenerationMIMIC3()
        >>> fn.task_name
        'CorGANGenerationMIMIC3'
    """

    task_name = "CorGANGenerationMIMIC3"
    input_schema = {"visits": "nested_sequence"}
    output_schema = {}
    _icd_col = "diagnoses_icd/icd9_code"

    def __call__(self, patient) -> List[Dict]:
        """Extract structured visit data for a single patient.

        Args:
            patient: A PyHealth patient object with admission and diagnosis
                event data.

        Returns:
            list of dict: A single-element list containing the patient record,
                or an empty list if the patient has fewer than 2 visits with
                diagnosis codes. Each dict has:
                    ``"patient_id"`` (str): the patient identifier.
                    ``"visits"`` (list of list of str): per-visit ICD code lists.
        """
        admissions = list(patient.get_events(event_type="admissions"))
        visits = []
        for adm in admissions:
            codes = (
                patient.get_events(
                    event_type="diagnoses_icd",
                    filters=[("hadm_id", "==", adm.hadm_id)],
                    return_df=True,
                )
                .select(pl.col(self._icd_col))
                .to_series()
                .drop_nulls()
                .to_list()
            )
            if codes:
                visits.append(codes)
        if len(visits) < 2:
            return []
        return [{"patient_id": patient.patient_id, "visits": visits}]


class CorGANGenerationMIMIC4(CorGANGenerationMIMIC3):
    """Task function for CorGAN synthetic EHR generation using MIMIC-IV.

    Inherits all logic from :class:`CorGANGenerationMIMIC3`. Overrides only
    the task name and the ICD code column to match the MIMIC-IV schema, where
    the column is ``icd_code`` (unversioned) rather than ``icd9_code``.

    Attributes:
        task_name (str): Unique task identifier.
        _icd_col (str): Polars column path for ICD codes in MIMIC-IV.

    Examples:
        >>> fn = CorGANGenerationMIMIC4()
        >>> fn.task_name
        'CorGANGenerationMIMIC4'
    """

    task_name = "CorGANGenerationMIMIC4"
    _icd_col = "diagnoses_icd/icd_code"


corgan_generation_mimic3_fn = CorGANGenerationMIMIC3()
corgan_generation_mimic4_fn = CorGANGenerationMIMIC4()
