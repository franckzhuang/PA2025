from datetime import datetime

class TrainingJobStore:
    def __init__(self, mongodb):
        self.collection = mongodb.db["training_jobs"]

    def create_job(self, job_id, model_type, params):
        job_doc = {
            "job_id": job_id,
            "model_type": model_type,
            "params": params,
            "status": "pending",
            "metrics": None,
            "error": None,
            "created_at": datetime.utcnow()
        }
        self.collection.insert_one(job_doc)

    def update_job(self, job_id, **kwargs):
        self.collection.update_one(
            {"job_id": job_id},
            {"$set": kwargs}
        )

    def get_job(self, job_id):
        return self.collection.find_one({"job_id": job_id}, {"_id": 0})
