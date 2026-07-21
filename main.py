from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, APIConnectionError, AuthenticationError, RateLimitError
from dotenv import load_dotenv
import os
import socket
import httpx

# ✅ 1. CONFIGURATION SYSTÈME (IMPORTANT POUR RENDER)
# Force la connexion directe sans passer par un proxy système
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# Charger les variables d'environnement (pour le local)
load_dotenv()

app = FastAPI(title="Pharmalink AI API")

# ✅ 2. CORS (Autoriser toutes les origines pour Flutter & Hoppscotch)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 3. CONFIGURATION OPENAI ROBUSTE
# Optimisé pour les environnements serverless comme Render
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
    timeout=30.0,          # Timeout de 30 secondes
    max_retries=3,         # Réessayer 3 fois en cas d'échec
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

# ✅ 4. MODÈLE DE REQUÊTE (Structure des données attendues)
class PharmaRequest(BaseModel):
    message: str
    patient_age: int = 30
    current_medications: list = []
    is_pregnant: bool = False
    is_breastfeeding: bool = False

# ✅ 5. PROMPT SYSTÈME (Règles médicales strictes)
SYSTEM_PROMPT = """
Tu es Pharmalink Advisor, un assistant pharmaceutique IA professionnel.

RÈGLES ABSOLUES :
1. ⚠️ URGENCE : Si le patient mentionne : douleur thoracique, difficulté respiratoire, perte de connaissance, saignement important, AVC → Réponds IMMÉDIATEMENT :
"🚨 URGENCE MÉDICALE
Ne prenez aucun médicament. Appelez immédiatement le 112 ou rendez-vous aux urgences les plus proches."

2. 🤰 FEMMES ENCEINTES/ALLAITANTES : Si is_pregnant=True ou is_breastfeeding=True → Toujours recommander de consulter un médecin avant tout médicament.

3. 👶 ENFANTS : Si patient_age < 12 ans → Recommander systématiquement de consulter un pédiatre ou pharmacien.

4. 💊 INTERACTIONS : Si current_medications contient plusieurs médicaments → Vérifier les interactions dangereuses.

5.  FORMAT DE RÉPONSE :
✅ Conseil (recommandation claire)
⚠️ Précautions (contre-indications, effets secondaires)
📞 Quand consulter (symptômes d'alerte)

6. Ne diagnoses JAMAIS. Oriente vers un professionnel de santé.
7. Base-toi sur les recommandations OMS/ANSM.
8. Sois concis (max 300 mots).
"""

# ✅ 6. ENDPOINT PRINCIPAL (API IA)
@app.post("/api/pharma-ai")
async def get_pharma_advice(req: PharmaRequest):
    try:
        # Vérification d'urgence locale (rapide)
        emergency_keywords = [
            "douleur thoracique", "coeur", "respiration", "étouffe", 
            "perdu connaissance", "saigne", "urgence", "112", "crise", "inconscient"
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

        # Construction du contexte patient pour l'IA
        patient_context = f"""
        **Profil du patient :**
        - Âge : {req.patient_age} ans
        - Médicaments actuels : {', '.join(req.current_medications) if req.current_medications else 'Aucun'}
        - Enceinte : {'Oui' if req.is_pregnant else 'Non'}
        - Allaite : {'Oui' if req.is_breastfeeding else 'Non'}
        
        **Question :** {req.message}
        """

        # Envoi de la requête à OpenAI
        print(f"📡 Envoi de la requête à OpenAI (Modèle: gpt-4o-mini)...")
        
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
        print(f"✅ Réponse reçue de l'IA avec succès")

        return {
            "advice": advice,
            "urgency_level": "low",
            "sources": ["OMS", "ANSM"]
        }

    # Gestion des erreurs spécifiques
    except AuthenticationError as e:
        print(f"❌ ERREUR AUTHENTIFICATION: Clé API invalide")
        raise HTTPException(status_code=401, detail="Clé API OpenAI invalide ou expirée")

    except RateLimitError as e:
        print(f"⚠️ LIMITE ATTEINTE: Trop de requêtes")
        raise HTTPException(status_code=429, detail="Limite de requêtes atteinte. Réessayez dans quelques minutes.")

    except APIConnectionError as e:
        print(f"❌ ERREUR CONNEXION: Impossible de joindre OpenAI")
        print(f"Détails: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service OpenAI indisponible: {str(e)}")

    except Exception as e:
        print(f"❌ Erreur inattendue: {type(e).__name__}")
        print(f"Détails: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

# ✅ 7. ENDPOINT DE DIAGNOSTIC RÉSEAU
@app.get("/debug/network")
async def debug_network():
    """
    Permet de tester si le serveur peut accéder à OpenAI et à internet.
    """
    import socket
    results = {}
    
    # Test 1 : DNS & TCP vers OpenAI
    try:
        socket.create_connection(("api.openai.com", 443), timeout=5)
        results["openai_tcp"] = "✅ OK (Connexion établie)"
    except Exception as e:
        results["openai_tcp"] = f"❌ Échec (Bloqué par le firewall ou DNS): {e}"
        
    # Test 2 : Requête HTTP basique
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get("https://api.openai.com/v1/models", 
                                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"})
            results["openai_http"] = f"✅ OK (Status: {resp.status_code})"
    except Exception as e:
        results["openai_http"] = f"❌ Échec (Impossible de contacter l'API): {str(e)}"
        
    # Test 3 : Vérification de la variable d'environnement
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        results["api_key_present"] = "✅ OK"
        results["api_key_preview"] = f"{api_key[:10]}..."
    else:
        results["api_key_present"] = "❌ MISSING"
        
    return results

# ✅ 8. ROUTE RACINE
@app.get("/")
def root():
    return {
        "message": "✅ API Pharmalink IA - Conseil Pharmaceutique",
        "status": "running",
        "endpoints": {
            "diagnosis": "/debug/network",
            "advisor": "/api/pharma-ai"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
