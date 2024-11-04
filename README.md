# 🌊 AQuaBot - AI Water Consumption Awareness Assistant

AQuaBot is an innovative AI assistant that combines powerful language understanding with real-time water consumption tracking. It aims to raise awareness about the environmental impact of AI models while providing helpful responses to user queries.

## 🌟 Try it Now!

Visit [AQuaBot on Hugging Face Spaces](https://huggingface.co/spaces/CamiloVega/aQuaBot/tree/main) to interact with the live demo!

## 🎯 Key Features

- **Real-time Water Tracking**: Monitors water consumption for each interaction
- **Interactive Chat Interface**: Built with Gradio for a seamless user experience
- **Environmental Impact Awareness**: Educates users about AI's water footprint
- **High-Performance Model**: Powered by Meta's Llama-2-7b model
- **Scientific Backing**: Calculations based on peer-reviewed research

## 🔧 Technical Stack

- **Model**: Meta-llama/Llama-2-7b-chat-hf
- **Frontend**: Gradio 4.19.2
- **Framework**: PyTorch 2.4.0
- **Dependencies**: Transformers, Accelerate, Huggingface-hub
- **Infrastructure**: Hugging Face Spaces with GPU acceleration

## 📊 Water Consumption Calculation

The application tracks water usage based on the research paper "Making AI Less Thirsty" (Li et al., 2023):

- Training water consumption per token: 0.0000309 ml
- Inference water consumption per token: 0.05 ml
- Real-time tracking for both input and output tokens

## 🚀 Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aquabot.git
cd aquabot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
```bash
export HUGGINGFACE_TOKEN='your_token_here'
```

4. Run the application:
```bash
python app.py
```

## 📖 Usage

1. Visit the [AQuaBot Space](https://huggingface.co/spaces/CamiloVega/aQuaBot/tree/main)
2. Type your question or message in the chat interface
3. Watch the water consumption meter as you interact with the bot
4. Use the "Clear Chat" button to reset the conversation and water counter

## 🧪 Requirements

See `requirements.txt` for detailed dependencies. Key requirements:
- transformers==4.36.2
- torch==2.4.0
- gradio==4.19.2
- accelerate==0.27.2

## 📚 Citation

```bibtex
@article{li2023making,
  title={Making AI Less Thirsty: Uncovering and Addressing the Secret Water Footprint of AI Models},
  author={Li, P. et al.},
  journal={ArXiv Preprint},
  year={2023},
  url={https://arxiv.org/abs/2304.03271}
}
```

## 👤 Author

**Camilo Vega Barbosa**
- AI Professor and Artificial Intelligence Solutions Consultant
- [LinkedIn](https://www.linkedin.com/in/camilo-vega-169084b1/)
- [GitHub](https://github.com/CamiloVga)

## ⚠️ Important Note

This implementation uses Meta's Llama-2-7b model instead of GPT-3 for accessibility and cost considerations. The water consumption calculations remain based on the research paper's findings.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/aquabot/issues).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with 💧 by [Camilo Vega](https://www.linkedin.com/in/camilo-vega-169084b1/)
