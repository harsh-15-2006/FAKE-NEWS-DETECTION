import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from config import TFIDF_MODEL_PATH, CLASSIFIER_MODEL_PATH

TRAINING_DATA = [
    ("BREAKING: UNESCO declares Indian National Anthem as the best in the world! Share this proud news immediately with all your friends!", 1),
    ("Drinking boiled garlic water cures coronavirus and all cancer in 24 hours. Doctors are hiding this miracle remedy from you!", 1),
    ("NASA warns of a massive solar storm next week that will permanently shut down all global internet for 6 months.", 1),
    ("URGENT: 5G mobile towers are releasing toxic radiation that kills birds and spreads airborne viruses. Forward immediately!", 1),
    ("PM Free Laptop Scheme: Click here to claim your free HP laptop. Only 500 units left! Forward to 10 WhatsApp groups.", 1),
    ("RBI Announcement: All currency notes with pen marks or writing are officially declared null and void from tomorrow. Banks will reject them!", 1),
    ("Miracle secret cure: Eat raw ginger and lemon peel twice daily to permanently eradicate diabetes and heart blockage without medication.", 1),
    ("Congratulations! You have won 25,00,000 INR in online international lottery. Send your bank details and OTP to claim your prize immediately.", 1),
    ("SHOCKING: Central government giving Rs 5000 directly into bank accounts for everyone who registers on this WhatsApp link!", 1),
    ("Scientists confirmed that drinking cold water after meals causes instant cancer. Spread this life saving truth to your family!", 1),
    ("அதிர்ச்சி செய்தி: தமிழக அரசு அனைத்து ரேஷன் அட்டைதாரர்களுக்கும் ரூ 5000 பொங்கல் ரொக்க பரிசு வழங்க உத்தரவிட்டுள்ளது! உடனே பகிருங்கள்!", 1),
    ("உங்கள் மின் இணைப்பு இன்று இரவு 9:30 மணிக்கு துண்டிக்கப்படும். உடனடியாக இந்த எண்ணை தொடர்பு கொள்ளவும். மின் கட்டணம் பாக்கி உள்ளது.", 1),
    ("அதிசய மூலிகை சாறு குடித்தால் ரத்த அழுத்தம் மற்றும் சர்க்கரை நோய் 3 நாட்களில் குணமாகும்! மருத்துவர்கள் மறைத்த உண்மை அம்பலம்!", 1),
    ("கலைஞர் மகளிர் உரிமை தொகை திட்டம் அதிரடியாக நிறுத்தப்படுகிறது. அரசு அவசர அறிவிப்பு.", 1),
    ("இலவச மொபைல் ரீசார்ஜ் 3 மாதங்களுக்கு பெற இந்த லிங்கை உடனே கிளிக் செய்து 10 நண்பர்களுக்கு பகிருங்கள்!", 1),
    ("Iniku night kulla indha link click panni 10 friends ku forward pannina free smartphone kedaikum. Government offer!", 1),
    ("EB bill pay panala nu iniku night current cut panna poranga. Udane indha number ku call panni verify pannunga scam!", 1),
    ("Vadhanthi illa unmai, boiled garlic water kudicha corona and cancer oru naal la poirum. Doctors namba vekkara poi!", 1),
    ("Free solar panel link click pannunga 100% free subsidy bank account la varum OTP kudunga.", 1),
    ("ISRO's Chandrayaan Pragyan rover confirms the unambiguous presence of Sulfur and other elements on lunar south pole using LIBS.", 0),
    ("The Reserve Bank of India clarified that notes with writing remain valid legal tender under the Clean Note Policy.", 0),
    ("Ministry of Finance released the monthly economic review highlighting GST revenue collections of 1.73 lakh crore for March.", 0),
    ("World Health Organization publishes updated clinical guidelines for hypertension management across South Asian populations.", 0),
    ("Tamil Nadu Government distributes official Pongal gift hampers containing 1 kg raw rice, 1 kg sugar, and whole sugarcane through PDS outlets.", 0),
    ("Indian Meteorological Department issues standard yellow alert for coastal districts due to cyclonic circulation over Bay of Bengal.", 0),
    ("IIT Madras researchers develop novel electrochemical catalyst to produce green hydrogen from seawater efficiently.", 0),
    ("TANGEDCO advises consumers to pay electricity bills only via official website tangedco.gov.in and warns against SMS fraud.", 0),
    ("Department of Telecommunications reassures citizens that non-ionizing RF emissions from telecom towers comply with stringent global safety guidelines.", 0),
    ("CBSE announces official schedule for class 10 and 12 board examinations commencing from February 15.", 0),
    ("தமிழ்நாடு மின் பகிர்மான கழகம்: மின் கட்டணங்களை நுகர்வோர்கள் அதிகாரப்பூர்வ இணையதளம் வாயிலாக மட்டுமே செலுத்த வேண்டும்.", 0),
    ("இஸ்ரோவின் சந்திரயான் விண்கலம் நிலவின் தென் துருவத்தில் கந்தகம் இருப்பதற்கான சான்றுகளை உறுதிப்படுத்தியுள்ளது என விஞ்ஞானிகள் தகவல்.", 0),
    ("தமிழகத்தில் வடகிழக்கு பருவமழை தீவிரமடைந்துள்ள நிலையில் முன்னெச்சரிக்கை நடவடிக்கைகளை அரசு தீவிரப்படுத்தியுள்ளது.", 0),
    ("மகளிர் உரிமைத் திட்டத்தின் கீழ் தகுதியுள்ள மகளிருக்கு மாதந்தோறும் ரூ 1000 அவரவர் வங்கி கணக்கில் நேரடியாக வரவு வைக்கப்படுகிறது.", 0),
    ("மருத்துவ நிபுணர்கள் கூற்றுப்படி டெங்கு காய்ச்சல் ஏற்பட்டால் சுய மருத்துவம் செய்யாமல் உடனடியாக மருத்துவமனை செல்ல வேண்டும்.", 0),
    ("ISRO moon mission pathi official website la update potrukanga rover successfully completed scientific experiments.", 0),
    ("Tamil Nadu govt official notification release pannirukanga pongal gift ration card holders ku ration kadaila kedaikum.", 0),
    ("RBI guidelines padi scribbled currency notes legal tender dhaan banks reject panna koodadhu.", 0),
    ("TNEB official website www.tangedco.gov.in la mattum bill pay pannunga nu advisory vandhuruku.", 0)
]

class FakeNewsMLClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(TFIDF_MODEL_PATH) and os.path.exists(CLASSIFIER_MODEL_PATH):
            try:
                self.vectorizer = joblib.load(TFIDF_MODEL_PATH)
                self.model = joblib.load(CLASSIFIER_MODEL_PATH)
                return
            except Exception as e:
                print(f"[ML] Error loading saved models: {e}. Re-training...")

        self.train(TRAINING_DATA)

    def train(self, data):
        texts = [x[0] for x in data]
        labels = [x[1] for x in data]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r'(?u)\b[\w\u0B80-\u0BFF]+\b'
        )
        X = self.vectorizer.fit_transform(texts)
        
        base_lr = LogisticRegression(C=2.0, class_weight='balanced', max_iter=200, random_state=42)
        self.model = CalibratedClassifierCV(estimator=base_lr, cv=3)
        self.model.fit(X, labels)

        os.makedirs(os.path.dirname(TFIDF_MODEL_PATH), exist_ok=True)
        joblib.dump(self.vectorizer, TFIDF_MODEL_PATH)
        joblib.dump(self.model, CLASSIFIER_MODEL_PATH)

    def predict(self, text: str) -> Dict[str, Any]:
        if not text or not self.vectorizer or not self.model:
            return {
                "ml_label": "Unverified",
                "fake_probability": 0.50,
                "real_probability": 0.50,
                "confidence": 0.50,
                "top_features": []
            }

        vec = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(vec)[0]
        real_prob = float(probabilities[0])
        fake_prob = float(probabilities[1])

        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = vec.toarray()[0]
        top_indices = np.argsort(tfidf_scores)[::-1][:6]
        top_features = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]

        if fake_prob >= 0.50:
            ml_label = "Fake / Misleading"
            confidence = fake_prob
        else:
            ml_label = "Real / Credible"
            confidence = real_prob

        return {
            "ml_label": ml_label,
            "fake_probability": round(fake_prob, 3),
            "real_probability": round(real_prob, 3),
            "confidence": round(confidence, 3),
            "top_features": top_features
        }

_ml_classifier_instance = None

def get_ml_classifier() -> FakeNewsMLClassifier:
    global _ml_classifier_instance
    if _ml_classifier_instance is None:
        _ml_classifier_instance = FakeNewsMLClassifier()
    return _ml_classifier_instance
