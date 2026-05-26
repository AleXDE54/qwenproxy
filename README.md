# Free **QwenAI** self-hosted API

## Features

- No QwenAI account needed!
- Works with all latest avaiable Qwen models
- History, Thinking intelegence working!
- Compatile with OpenAI API
- G4F-free!!

## Instalation

### Docker Method

```bash
docker build -t qwenproxy .
docker run -d -p 1234:1234 --name qwenproxy qwenproxy
```

### Manual method
Clone the repo
```bash
git clone https://github.com/AleXDE54/qwenproxy.git
```

Install the requirements
```bash
pip install --break-system-packages -r req.txt
```

Run the python file
```bash
nohup python qwenproxy.py &
```
