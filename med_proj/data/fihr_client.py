import requests
from typing import Dict, Any, Optional
from care_bounce.common.logging import get_logger

log = get_logger("fhir_client")

class FHIRClient:
    def __init__(self, base_url: str, bearer_token: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if bearer_token:
            self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})
        self.session.headers.update({"Accept": "application/fhir+json"})

    def get(self, resource: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{resource}"
        r = self.session.get(url, params=params, timeout=self.timeout)
        if r.status_code >= 400:
            log.error("FHIR GET failed %s %s -> %s %s", url, params, r.status_code, r.text[:300])
            r.raise_for_status()
        return r.json()

    def iter_bundle(self, resource: str, params: Dict[str, Any], max_pages: int = 50):
        # basic paging: follow 'next' links if present
        bundle = self.get(resource, params)
        yield bundle
        pages = 1
        while pages < max_pages:
            next_url = None
            for link in bundle.get("link", []):
                if link.get("relation") == "next":
                    next_url = link.get("url")
                    break
            if not next_url:
                return
            r = self.session.get(next_url, timeout=self.timeout)
            if r.status_code >= 400:
                r.raise_for_status()
            bundle = r.json()
            pages += 1
            yield bundle