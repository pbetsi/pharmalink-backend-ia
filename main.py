from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, APIConnectionError, AuthenticationError
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

app = FastAPI(title="Pharmalink AI API")

# ✅ Activer CORS (pour autoriser Flutter Web et Hoppscotch)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Configuration OpenAI avec TIMEOUT et RETRIES (Crucial pour Render)
# Si la connexion échoue, il réessaiera 2 fois et attendra max 30 secondes
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,  # Timeout de 30 secondes
    max_retries=2  # Réessayer 2 fois en cas d'échec
)

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

4. 💊 INTERACTIONS : Si current_medications contient plusieurs médicaments → Vérifier les interactions dangereuses.

5. 📋 FORMAT DE RÉPONSE :
✅ Conseil (recommandation claire)
⚠️ Précautions (contre-indications, effets secondaires)
📞 Quand consulter (symptômes d'alerte)

6. Ne diagnoses JAMAIS. Oriente vers un professionnel de santé.
7. Base-toi sur les recommandations OMS/ANSM.
8. Sois concis (max 300 mots).
"""

@app.post("/api/pharma-ai")
async def get_pharma_advice(req: PharmaRequest):
    try:
        # ✅ 1. Vérification urgence (mots-clés)
        emergency_keywords = [
            "douleur thoracique", "coeur", "respiration", "étouffe", 
            "perdu connaissance", "saigne", "urgence", "112", "crise", "inconscient"
        ]
        
        if any(kw in req.message.lower() for kw in emergency_keywords):
            return {
                "advice": "🚨 **URGENCE MÉDICALE DÉTECTÉE**\n\n"
                         "Ne prenez aucun médicament.\n"
                         "Appelez immédiatement le **112** ou rendez-vous aux urgences.\n\n"
                         "**Numéros d'urgence :**\n"
                         "🚑 SAMU : 112 (gratuit)\n"
                         "🏥 Urgences 24h/24",
                "urgency_level": "critical",
                "sources": []
            }

        # ✅ 2. Construction du contexte patient
        patient_context = f"""
        **Profil du patient :**
        - Âge : {req.patient_age} ans
        - Médicaments actuels : {', '.join(req.current_medications) if req.current_medications else 'Aucun'}
        - Enceinte : {'Oui' if req.is_pregnant else 'Non'}
        - Allaite : {'Oui' if req.is_breastfeeding else 'Non'}
        
        **Question :** {req.message}
        """

        # ✅ 3. Appel à OpenAI (Modèle GPT-4o-mini pour rapidité/coût)
        print(f"📡 Envoi de la requête à OpenAI...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": patient_context}
            ],
            temperature=0.3,
            max_tokens=500
        )

        advice = response.choices[0].message.content
        print(f"✅ Réponse reçue de l'IA")

        return {
            "advice": advice,
            "urgency_level": "low",
            "sources": ["OMS", "ANSM"]
        }

    # ✅ GESTION DES ERREURS SPÉCIFIQUES
    except AuthenticationError as e:
        print(f"❌ ERREUR AUTHENTIFICATION: Clé API invalide")
        raise HTTPException(status_code=401, detail="Clé API OpenAI invalide ou expirée")

    except APIConnectionError as e:
        print(f"❌ ERREUR CONNEXION: Impossible de joindre OpenAI")
        print(f"Détails: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service OpenAI indisponible: {str(e)}")

    except Exception as e:
        print(f"❌ Erreur inattendue: {type(e).__name__}")
        print(f"Détails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@app.get("/")
def root():
    return {"message": "✅ API Pharmalink IA - Conseil Pharmaceutique"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
