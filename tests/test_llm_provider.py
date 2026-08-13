import io
import json
import unittest
from unittest.mock import patch

from llm_provider import OllamaProvider


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class OllamaProviderTests(unittest.TestCase):
    @patch("llm_provider.urlopen")
    def test_calls_non_streaming_local_generate_api(self, urlopen):
        urlopen.return_value = FakeResponse(json.dumps({"response": " answer "}).encode())

        answer = OllamaProvider(model="gemma3").generate("prompt")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(payload, {
            "model": "gemma3",
            "prompt": "prompt",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        })
        self.assertEqual(answer, "answer")


if __name__ == "__main__":
    unittest.main()
