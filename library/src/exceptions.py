class DhwaniAPIError(Exception):
    def __init__(self, response):
        super().__init__(f"API Error {response.status_code}: {response.text}")
        self.status_code = response.status_code
        self.response = response
