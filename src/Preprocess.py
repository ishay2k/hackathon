import pandas as pd
import numpy as np
import re
from datetime import datetime


class Preprocess:

    def __init__(self, filepath1, filepath2, filepath3):
        self.__data = pd.read_csv(filepath1, encoding="utf-8")
        self.__labels_0 = pd.read_csv(filepath2)
        self.__labels_1 = pd.read_csv(filepath3)

        self.clean_column_names()
        self.drop_id()
        self.clean_age()
        self.clean_her2()
        self.simplify_histological_diagnosis()
        self.clear_histological_degree()
        self.clean_ivi_lymphovascular_invasion()
        self.clear_KI67()
        self.delete_lymphatic_penetration()
        self.clean_T_column()
        self.clean_M_column()
        self.clean_N_column()
        self.fix_margin_type()

        self.preProcess()

    def clean_column_names(self):
        """
        מנקה את שמות העמודות כך שנשאר רק החלק באנגלית (אם יש).
        לדוגמה: 'אבחנה-T -Tumor mark (TNM)' → 'Tumor mark (TNM)'
        """
        new_columns = []
        for col in self.__data.columns:
            # מנסה למצוא את החלק באנגלית
            match = re.search(r'[-–]\s*([A-Za-z].*)$', col)
            if match:
                new_columns.append(match.group(1).strip())
            else:
                # אם אין אנגלית – מסיר רווחים מיותרים
                new_columns.append(col.strip())
        self.__data.columns = new_columns

    def drop_id(self):
        """
        dropping id's
        """
        self.__data.drop(["hushed_internalpatientid"], axis=1, inplace=True)

    def clean_age(self):
        """
        cleans the age column
        Returns
        -------
        """
        age_col = self.__data["Age"]
        valid_ages = age_col[(age_col >= 0) & (age_col <= 120)]
        mean_age = valid_ages.mean()

        self.__data.loc[(age_col < 0) | (age_col > 120), "Age"] = mean_age

    def clean_her2(self):
        """
        Cleans the HER2 column and maps messy entries to:
        - 'positive'
        - 'negative'
        - 'equivocal'
        Unknown or unclassified values are set to NaN.
        """

        def classify_her2(value):
            if pd.isnull(value):
                return "unknown"

            val = str(value).strip().lower()

            # --- Positive ---
            positive_keywords = [
                'positive', 'amplified', '+3', 'pos', 'fisher amplified',
                'positive by ihc and fish', 'her 2 +3', 'her 2 pos', 'score 1',
                '(+)', '(+3)', '(+)3', '(+1)', '+@', "+ratio 2'3", '1+/10%', '1+',
                'po', 'posit', '+', '1', '2', '3', 'FISH POS', '+2 FISH-pos', '+2 FISH(-)',
                '/NEU-POSITIVE+3', '2+', '+2, FISH חיובי', 'POSITIVE +3', 'FISH+', 'Pos. FISH=2.9',
                'positive. FISH amplified 2.3', 'Positive by FISH', 'POS +3', '+1, +2 FISH pos at surgery',
                '+2 IHC', '+3 100%cells', 'FISH (-)', 'positive', 'POS', '+2 FISH AMPLIFIED', 'Pos by FISH',
                'Pos. FISH 2.26', '+2 FISH pos ratio 2.8', '+3 100%', '+2 FISH +', 'AMPLIFIED',
                'Positive. FISH 5.75', '+3 (100%cells)', '2, ordered fish 1.2', '0-1', '+2 FISH amplified',
                '+2 FISH positive', '+2 FISHpos', '+2 in bx FISH+ ratio 2.05', '+2 FISH+' 'FISH pos',
                'positive fish 2,2',
                'FISH amplified 3.3', 'Posit', 'Pos',
            ]

            # --- Negative ---
            negative_keywords = [
                'negative', 'neg', 'non amplified', 'not amplified',
                'negative bu ihc and fish', 'no', 'negarive', 'negatie',
                'akhkh', 'akhah', 'heg', 'heg', 'meg', 'nfg', 'nef', 'nd',
                '1+/20% 9negative', ',eg', 'שלילי', 'NEGATIVE PER FISH', 'Neg', 'neg', '-',
                'FISH -', '(-)', '2 non amplified', '+2 Fish NEG', 'Negative', 'negative',
                'fish neg', 'NEGATIVE', '+2 FISH-neg', 'HER2/NEU-NEG', 'Neg ( FISH non amplified)',
                'FISH neg', 'negative by FISH', 'Neg vs +2', '/NEU-NEG', 'neg.', '+2 FISH negative',
                'Neg by IHC and FISH', '2 fish non amplified', 'NEG PER FISH', '/NEU- NEG', '2 not amplified',
                'Neg( FISH Non amplified)', '/NEU-NEGATIV', 'Neg by FISH', 'FISH Non Amplified', '+2 FISH-',
                '(-) by FISH', '2 FISH: NOT AMPLIFIED', 'indeterminate, FISH neg', 'NEG by FISH 1.0',
                'neg by FISH (1.14)',
                'FISH NEG', 'HER2- (-)', '0 neg',

            ]

            # --- Equivocal ---
            equivocal_keywords = [
                'equivocal', 'intermediate', 'indeterminate', 'pending', '--', '=', '0', 'indeterm',
            ]

            # Check for keywords
            if any(k in val for k in positive_keywords):
                return 'positive'
            elif any(k in val for k in negative_keywords):
                return 'negative'
            elif any(k in val for k in equivocal_keywords):
                return 'equivocal'
            else:
                return "unknown"  # unclassified

        self.__data['Her2'] = self.__data['Her2'].apply(classify_her2)

    def simplify_histological_diagnosis(self):
        # Most common malignant types - keep as-is
        keep_as_is = {
            'INFILTRATING DUCT CARCINOMA',
            'LOBULAR INFILTRATING CARCINOMA',
            'INTRADUCTAL CARCINOMA',
            'INFILTRATING DUCTULAR CARCINOMA WITH DCIS',
            'DUCTAL AND LOBULAR CARCINOMA',
        }

        other = {
            'TUBULAR CARCINOMA',
            'LOBULAR CARCINOMA IN SITU',
            'DUCTAL CARCINOMA IN SITU',
            'MEDULLARY CARCINOMA',
            'MUCINOUS ADENOCARCINOMA',
            'PAGET`S DISEASE OF BREAST',
            'PAPILLARY CARCINOMA',
            'INFLAMMATORY CARCINOMA',
            'COMEDOCARCINOMA',
            'INTRADUCTAL PAPILLOMA',
            'ADENOMA OF NIPPLE',
            'INTRACYSTIC PAP ADENOMA',
            'PHYLLODES TUMOR MALIGNANT',
            'PHYLLODES TUMOR BENIGN',
            'PHYLLODES TUMOR NOS'
        }

        # Malignant but not otherwise specified or ambiguous
        malignant_nos = {
            '"TUMOR  MALIGNANT, NOS"',
            '"CARCINOMA, NOS"',
            'INTRADUCTAL PAP CARCINOMA WITH INVASION',
            'INTRADUCT AND LOBULAR CARCINOMA IN SITU',
            'PAGET`S AND INTRADUCTAL CARCINOMA OF BREAST',
            'ADENOCARCINOMA',
            'COMEDOCARCINOMA IN SITU'
        }

        # Rare malignancies
        rare_malignancy = {
            'NEUROENDOCRINE CARCINOMA',
            'MUCIN PRODUCING ADENOCARCINOMA',
            'APOCRINE ADENOCARCINOMA',
            '"ADENOID CYSTIC CA,ADENOCYSTIC CA"',
            'PAPILLARY ADENOCARCINOMA',
            'INTRADUCTAL PAPILLARY CARCINOMA',
            'INTRACYSTIC CARCINOMA',
            '"VERRUCOUS CARCINOMA, VERRUCOUS SQUAMOUS CELL CARC'
        }

        # Clearly benign NOS
        benign_nos = {
            '"BENIGN TUMOR, NOS"',
            '"FIBROADENOMA, NOS"',
            '"INTRADUCTAL PAPILLOMATOSIS, NOS"'
        }

        def map_diagnosis(val):
            if val in keep_as_is:
                return val
            elif val in malignant_nos:
                return 'Malignant, NOS'
            elif val in rare_malignancy:
                return 'Rare malignancy'
            elif val in benign_nos:
                return 'Benign, NOS'
            elif val in other:
                return 'other'
            else:
                return 'Other/Unknown'

        self.__data['Histological diagnosis'] = self.__data['Histological diagnosis'].apply(map_diagnosis)

    def clear_histological_degree(self):
        """
        considering G4 appears only once, we will add it to G3 so we won't overfit
        """
        self.__data['Histopatological degree'] = self.__data['Histopatological degree'].replace({
            'G4 - Undifferentiated': 'G3 - Poorly differentiated'
        })

    def clean_ivi_lymphovascular_invasion(self):
        """
        column will be deleted because of too many blank lines
        """
        # self.__data = self.__data.drop('Ivi -Lymphovascular invasion', axis=1)

    def clear_KI67(self):
        """
        turn the column to three different classes
        """

        def classify_ki67(value):
            if pd.isna(value):
                return 'Unknown'

            value = str(value).lower().strip()

            # Handle clear textual clues
            if 'low' in value:
                return 'Low'
            if 'high' in value or 'score iv' in value or '>50%' in value or '>75%' in value or '>90%' in value:
                return 'High'
            if 'intermediate' in value or 'int' in value or 'score 2-3' in value:
                return 'Intermediate'
            if 'negative' in value or 'no' in value:
                return 'Low'

            # Extract all numbers from the string (could be percentages or counts)
            numbers = re.findall(r'\d+\.?\d*', value)
            if not numbers:
                return 'Unknown'

            # Convert extracted numbers to floats
            numbers = [float(num) for num in numbers]

            # Use the first number as representative value
            ki67_num = numbers[0]

            # Classification based on numeric thresholds
            if ki67_num < 10:
                return 'Low'
            elif 10 <= ki67_num <= 20:
                return 'Intermediate'
            elif ki67_num > 20:
                return 'High'
            else:
                return 'Unknown'

        self.__data['KI67 protein'] = self.__data['KI67 protein'].apply(classify_ki67)

    def delete_lymphatic_penetration(self):
        self.__data.drop(["Lymphatic penetration"], axis=1, inplace=True)

    def clean_T_column(self):
        """
        Simplifies unique 'T' values in the TNM staging system by mapping subcategories
        (like 'T2a') to their main stage ('T2').
        """
        t_col = self.__data["T -Tumor mark (TNM)"]
        unique_values = t_col.unique()

        # Create a mapping dictionary
        value_map = {}
        for value in unique_values:
            if pd.isna(value) or value == "Not yet Established":
                value_map[value] = value  # Leave unchanged or handle later
            elif isinstance(value, str) and value.startswith("T") and len(value) > 2:
                for i in range(2, len(value)):
                    if not value[i].isdigit():
                        value_map[value] = value[:i]  # Simplify 'T2a' to 'T2'
                        break
                else:
                    value_map[value] = value  # Keep as-is if no letter found
            else:
                value_map[value] = value  # Keep normal values as-is

        # Replace values in the column using the map
        self.__data["T -Tumor mark (TNM)"] = t_col.map(value_map)

    def clean_M_column(self):
        """
        Cleans the 'M' column in the TNM system. Keeps 'M0', 'M1', 'MX' as-is.
        Simplifies values like 'M1a', 'M1b' to 'M1'.
        Leaves NaN and 'Not yet Established' for now.
        """

        def simplify_m_stage(value):
            if pd.isna(value) or value == "Not yet Established":
                return value  # optional: return "Unknown"
            if isinstance(value, str):
                if re.fullmatch(r"M[01X]", value):  # Keep M0, M1, MX
                    return value
                elif re.match(r"M1", value):  # Anything that starts with M1 → M1
                    return "M1"
            return value  # Default: unchanged

        self.__data["M -metastases mark (TNM)"] = (
            self.__data["M -metastases mark (TNM)"].apply(simplify_m_stage))

    def clean_N_column(self):
        valid_prefixes = ('N0', 'NX', 'ITC', 'Not yet Established')

        def filter_n(value):
            if pd.isna(value):
                return value  # keep NaN
            if isinstance(value, str):
                if value.startswith(('N1', 'N2', 'N3')):
                    return value
                if value in valid_prefixes:
                    return value
                # Set invalid ones to NaN (like N4)
                return np.nan
            return value

        self.__data["N -lymph nodes mark (TNM)"] = self.__data["N -lymph nodes mark (TNM)"].apply(filter_n)

        # Now drop rows where 'N' is '#NAME?'
        valid_indices = self.__data[self.__data["N -lymph nodes mark (TNM)"] != "#NAME?"].index

        # Filter all three structures using the same valid indices
        self.__data = self.__data.loc[valid_indices].copy()
        self.__labels_0 = self.__labels_0.loc[valid_indices].copy()
        self.__labels_1 = self.__labels_1.loc[valid_indices].copy()

    def fix_margin_type(self):
        mapping = {
            'נקיים': 'clean',
            'ללא': 'without',
            'נגועים': 'contaminated'
        }

        self.__data['Margin Type'] = self.__data['Margin Type'].map(mapping)

    def average_of_valid_numeric_values(self, column_name):
        """
        Returns the average of valid numeric values in the specified column.
        Valid values: numeric and not NaN.
        """
        # Convert values to numeric; non-numeric entries will become NaN
        numeric_values = pd.to_numeric(self.__data[column_name], errors='coerce')

        # Drop NaN values to keep only valid numeric entries
        valid_values = numeric_values.dropna()

        # Return None if no valid values exist
        if valid_values.empty:
            return None

        # Return the mean of valid numeric values
        return valid_values.mean()

    def fill_invalid_numeric_values(self, column_name, fill_value):
        """
        Fill invalid (non-numeric or NaN) values in a numeric column with a given value.
        """
        self.__data[column_name] = pd.to_numeric(self.__data[column_name], errors='coerce')  # Convert to numeric
        self.__data[column_name] = self.__data[column_name].fillna(fill_value)  # Safe assignmentlid to NaN

    def validate_numeric_column(self, invalid_value=np.nan):  # TODO
        """
        מאמתת שעבור כל ערך בעמודת 'Age', הוא בתחום [0, 120].
        ערכים שמחוץ לתחום יוחלפו ב־invalid_value.
        פועלת ישירות על self.__data.
        """

        # Age
        self.__data['Age'] = pd.to_numeric(self.__data['Age'], errors='coerce')

        # החלפת ערכים לא תקינים
        self.__data.loc[
            (self.__data['Age'] < 0) | (self.__data['Age'] > 120),
            'Age'
        ] = invalid_value

    def map_er_category(self, value):
        val = str(value).lower().strip()

        if any(neg in val for neg in ["-", "neg", "negative"]) or re.match(r"^-\d", val):
            return -1
        if any(pos in val for pos in ["+", "positive", "pos", "strong", "100", "++"]) or re.search(r"\d+%.*pos", val):
            return 1
        if val in ["", "unknown", "?", "#name?", "pop", "nge"]:
            return 0
        try:
            num_val = float(val.replace(",", "."))
            if num_val < 1:
                return 0
        except:
            pass
        return 0

    def map_pr_category(self, value):
        if pd.isnull(value) or str(value).strip() == "" or value in ["unknown", "?", "#NAME?", "NEGNEG", "NEG/POS",
                                                                     "POSNEG"]:
            return 0  # unknown

        value = str(value).lower()

        # Positive indicators
        positive_keywords = ["pos", "+", "positive", "strong", "3+", "2+", "1+", "++", "+++", "strongly", "100%",
                             "weakly pos", "weak pos", "moderate", "intermediate"]
        if any(keyword in value for keyword in positive_keywords):
            return 1

        # Negative indicators
        negative_keywords = ["neg", "-", "negative", "שלילי"]
        if any(keyword in value for keyword in negative_keywords):
            return -1

        # Default to unknown
        return 0

    def map_actual_activity_category(self, value):
        """
        Maps raw 'Actual activity' descriptions to unified surgical categories:
        ----------
        value : str
            Raw string describing the actual activity (surgery)

        Returns:
        -------
        str
            A normalized surgical category
        """
        if pd.isnull(value):
            return "Unknown"

        val = str(value).strip()

        if val in ["unknown", "", "לא ידוע"]:
            return "Unknown"

        val = val.replace(" ", "").lower()

        # Category: Lumpectomy + Nodes
        lumpectomy_keywords = [
            "למפקטומי", "שד-כריתהבגישהזעירה", "גישהזעירה", "דרךהעטרה", "intrabeam"
        ]
        if "למפקטומי" in val or any(key in val for key in lumpectomy_keywords):
            return "Lumpectomy + Nodes"

        # Category: Mastectomy + Nodes
        if "מסטקטומי" in val:
            return "Mastectomy + Nodes"

        # Category: Axillary Surgery
        if "בלוטות" in val and "שד" not in val:
            return "Axillary Surgery"

        # Category: Anything else
        return "Other"

    def months_since_date(self, date_value):
        """
        Calculates how many full months have passed since the given date.

        Parameters
        ----------
        date_value : str or datetime
            The original date to calculate from.
        today : datetime, optional
            The reference date (default: current date)

        Returns
        -------
        int or None
            Number of full months since the date, or None if input is invalid
        """
        today = datetime.today()

        try:
            # Try to parse the date if it's a string
            if isinstance(date_value, str):
                date = pd.to_datetime(date_value, errors='coerce')
            else:
                date = pd.to_datetime(date_value)

            if pd.isnull(date):
                return None

            months = (today.year - date.year) * 12 + (today.month - date.month)
            if today.day < date.day:
                months -= 1  # not yet a full month

            return max(0, months)

        except Exception:
            return None

    def preProcess(self):

        # Diagnosis date
        # self.__data["Diagnosis date"] = self.__data["Diagnosis date"].apply(self.months_since_date())

        # Nodes exam
        avg = self.average_of_valid_numeric_values("Nodes exam")
        self.fill_invalid_numeric_values("Nodes exam", avg)

        # Positive nodes
        avg = self.average_of_valid_numeric_values("Positive nodes")
        self.fill_invalid_numeric_values("Positive nodes", avg)

        # Side
        self.__data["Side"] = self.__data["Side"].replace("ימין", "right")
        self.__data["Side"] = self.__data["Side"].replace("שמאל", "left")
        self.__data["Side"] = self.__data["Side"].replace("דו צדדי", "both")
        self.__data = self.__data.fillna("unknown")

        # stage
        self.__data["Stage"] = self.__data["Stage"].replace("Not yet Established", "unknown")
        self.__data["Stage"] = self.__data["Stage"].replace("LA", "unknown")
        self.__data["Stage"] = self.__data["Stage"].replace("Stage1c", "Stage1")
        self.__data["Stage"] = self.__data["Stage"].astype(str).str.replace("Stage", "", regex=False).str.strip()
        # todo calculate stage of unknown

        # Surgery date removal
        self.__data = self.__data.drop("Surgery date1", axis=1)
        self.__data = self.__data.drop("Surgery date2", axis=1)
        self.__data = self.__data.drop("Surgery date3", axis=1)

        # Surgery names
        self.__data["Surgery name1"] = self.__data["Surgery name1"].astype(str).str.replace(
            "AXILLARY DISSECTION",
            "AXILLARY LYMPH NODE DISSECTION",
            regex=False
        )
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace(
            "LUMPECTOMY / LOCAL EXCISION OF LESION OF BREAST", "LUMPECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("BILATERAL SIMPLE MASTECTOMY", "MASTECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("SUBTOTAL MASTECTOMY", "MASTECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("UNILATERAL SIMPLE MASTECTOMY",
                                                                            "MASTECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("SIMPLE EXCISION OF AXILLARY LYMPH NODE",
                                                                            "AXILLARY LYMPH NODE DISSECTION")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("RADICAL EXCISION OF AXILLARY LYMPH NODES",
                                                                            "AXILLARY LYMPH NODE DISSECTION")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("AXILLARY LYMPH NODE DISSECTION",
                                                                            "AXILLARY LYMPH NODE DISSECTION")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("BILATERAL EXTENDED RADICAL MASTECTOMY",
                                                                            "BILATERAL MASTECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("BILATERAL EXTENDED SIMPLE MASTECTOMY",
                                                                            "BILATERAL MASTECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("BILATERAL RADICAL MASTECTOMY",
                                                                            "BILATERAL MASTECTOMY")

        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("QUADRANTECTOMY", "LUMPECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("SOLITARY LYMPH NODE BIOPSY",
                                                                            "LYMPH NODE BIOPSY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("BIOPSY OF LYMPHATIC STRUCTURE",
                                                                            "LYMPH NODE BIOPSY")  # TODO
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("BILATERAL RADICAL MASTECTOMY",
                                                                            "BILATERAL MASTECTOMY")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("LYMPH NODE BIOPSY", "unknown")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("EXCISION OF ECTOPIC BREAST TISSUE",
                                                                            "unknown")
        self.__data["Surgery name1"] = self.__data["Surgery name1"].replace("BILATERAL SALPINGO-OOPHORECTOMY",
                                                                            "unknown")
        self.__data = self.__data.drop("Surgery name2", axis=1)
        self.__data = self.__data.drop("Surgery name3", axis=1)

        # Tumor depth - deleting because only 4 samples with data
        self.__data = self.__data.drop("Tumor depth", axis=1)

        # Tumor width - deleting because only 250 samples with
        self.__data = self.__data.drop("Ivi -Lymphovascular invasion", axis=1)

        # er - converting to 1,0,-1 from positive, unknown, negative
        self.__data["er"] = self.__data["er"].apply(self.map_er_category)

        # pr
        self.__data["pr"] = self.__data["pr"].apply(self.map_pr_category)

        # surgery before or after-Actual activity
        self.__data = self.__data.drop("Activity date", axis=1)

        # Actual activity
        self.__data["Actual activity"] = self.__data["Actual activity"].apply(self.map_actual_activity_category)

    def get_data(self):
        return self.__data

    def get_labels_0(self):
        return self.__labels_0

    def get_labels_1(self):
        return self.__labels_1


if __name__ == '__main__':
    data = Preprocess(r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.feats.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.0.csv",
                      r"C:\Users\hilib\PycharmProjects\IML\hackathon\hackathon\train_test_splits\train.labels.1.csv"
                      )