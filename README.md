<div align="center"><img width="300" alt="BirdNET+ logo" src="public/img/birdnet-logo-circle.png"></div>

# real-time-pwa (BirdNET Live)

**BirdNET Live** is a Progressive Web App (PWA) that brings the power of the BirdNET acoustic identification algorithm directly to your web browser. It allows you to identify bird species from their sounds in real-time using your device's microphone (requires microphone permission).

Key features:
*   **Run offline**: All processing happens locally on your device using TensorFlow.js. No audio data is ever uploaded to a server (models and assets are downloaded once and stored locally).
*   **Real-Time Identification**: Visualizes sound via a spectrogram and provides instant species predictions.
*   **Location-Aware**: Uses your device's geolocation (optional) to filter predictions for species likely to be found in your area.
*   **Offline Capable**: Once loaded, the app works without an internet connection.
*   **Cross-Platform**: Runs on desktop and mobile browsers (Chrome, Safari, Firefox, Edge).

⚠️ **Note**: This project is still in active development. Features and performance may vary across devices and browsers.

## Usage

You can access the live version of this project at: [https://birdnet-team.github.io/real-time-pwa/](https://birdnet-team.github.io/real-time-pwa/)

To install the PWA on your device, open the site in a compatible browser (e.g., Chrome, Edge, Firefox) and follow the prompts to add it to your home screen or desktop (choose "Install App" when prompted).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/birdnet-team/real-time-pwa.git
   cd real-time-pwa
   ```

2. Build and run the site locally:
   ```bash
   npm install
   npm run serve
   ```
3. Open your browser and navigate to `http://localhost:8080` to view the site.

## License

- **Source Code**: The source code for this project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
- **Models**: The models used in this project are licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

Please ensure you review and adhere to the specific license terms provided with each model.

## Citation

Feel free to use BirdNET for your acoustic analyses and research. If you do, please cite as:

```bibtex
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

## Funding

Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
