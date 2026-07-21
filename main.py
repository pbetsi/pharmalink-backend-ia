from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Pharmalink AI API")

# ✅ Activer CORS (pour autoriser Flutter Web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, remplacez par votre domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Configuration OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Modèle de requête
class PharmaRequest(BaseModel):
    message: str
    patient_age: int = 30
    current_medications: list = []
    is_pregnant: bool = False
    is_breastfeeding: bool = False

# ✅ Prompt système médical STRICT
SYSTEM_PROMPT = """
Tu es Pharmalink Advisor, un assistant pharmaceutique IA professionnel.

RÈGLES ABSOLUES :
1. ⚠️ URGENCE : Si le patient mentionne : douleur thoracique, difficulté respiratoire, perte de connaissance, saignement important, AVC → Réponds IMMÉDIATEMENT :
"🚨 URGENCE MÉDICALE
Ne prenez aucun médicament. Appelez immédiatement le 112 ou rendez-vous aux urgences les plus proches."

2. 🤰 FEMMES ENCEINTES/ALLAITANTES : Si is_pregnant=True ou is_breastfeeding=True → Toujours recommander de consulter un médecin avant tout médicament.

3. 👶 ENFANTS : Si patient_age < 12 ans → Recommander systématiquement de consulter un pédiatre ou pharmacien.

4.  INTERACTIONS : Si current_medications contient plusieurs médicaments → Vérifier les interactions dangereuses.

5. 📋 FORMAT DE RÉPONSE :
✅ Conseil (recommandation claire)
️ Précautions (contre-indications, effets secondaires)
📞 Quand consulter (symptômes d'alerte)

6. Ne diagnoses JAMAIS. Oriente vers un professionnel de santé.
7. Base-toi sur les recommandations OMS/ANSM.
8. Sois concis (max 300 mots).
"""

@app.post("/api/pharma-ai")
async def get_pharma_advice(req: PharmaRequest):
    try:
        # ✅ Vérification urgence (mots-clés)
        emergency_keywords = [
            "douleur thoracique", "coeur", "respiration", "étouffe", 
            "perdu connaissance", "saigne", "urgence", "112", "crise"
        ]
        
        if any(kw in req.message.lower() for kw in emergency_keywords):
            return {
                "advice": " **URGENCE MÉDICALE DÉTECTÉE**\n\n"
                         "Ne prenez aucun médicament.\n"
                         "Appelez immédiatement le **112** ou rendez-vous aux urgences.\n\n"
                         "**Numéros d'urgence :**\n"
                         "🚑 SAMU : 112 (gratuit)\n"
                         " Urgences 24h/24",
                "urgency_level": "critical",
                "sources": []
            }

        # ✅ Construction du contexte patient
        patient_context = f"""
        **Profil du patient :**
        - Âge : {req.patient_age} ans
        - Médicaments actuels : {', '.join(req.current_medications) if req.current_medications else 'Aucun'}
        - Enceinte : {'Oui' if req.is_pregnant else 'Non'}
        - Allaite : {'Oui' if req.is_breastfeeding else 'Non'}
        
        **Question :** {req.message}
        """

        # ✅ Appel à OpenAI GPT-4o-mini (économique)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # ~0.01€ pour 100 questions
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": patient_context}
            ],
            temperature=0.3,  # Faible pour plus de précision
            max_tokens=500
        )

        advice = response.choices[0].message.content

        return {
            "advice": advice,
            "urgency_level": "low",
            "sources": ["OMS", "ANSM"]  # À améliorer avec RAG
        }

    except Exception as e:
        # ✅ CORRECTION : Indentation correcte ici (4 espaces)
        print(f" Erreur IA : {e}")
        print(f"❌ Type : {type(e).__name__}")
        print(f" Clé API présente : {bool(os.getenv('OPENAI_API_KEY'))}")
        print(f" Détails complets : {str(e)}")
        
        # Message d'erreur plus détaillé
        error_detail = str(e)
        raise HTTPException(status_code=500, detail=f"Erreur IA: {error_detail}")

@app.get("/")
def root():
    return {"message": " API Pharmalink IA - Conseil Pharmaceutique"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
