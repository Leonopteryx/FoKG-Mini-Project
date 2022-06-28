# FoKG-Mini-Project
The Carcinogenesis classification practice. Foundation of knowledge graph. University of Paderborn.


## Build by: Group V

* Alkid Baci
* Harshal Tarmale 
* Iman Khastkhodaei

## Running Using Docker
1. Clone the repository
   ```sh
   git clone https://github.com/Leonopteryx/FoKG-Mini-Project.git
   cd .\FoKG-Mini-Project\
   ```
2. Build and run the Docker
   ```sh
   docker build -t fokg .
   docker run fokg
   ```
3. Copy the Classification results to your directory
   ```sh
   docker cp <container_id>:/classification_result.ttl ./FoKG-Mini-Project/
   ```
   * For Container id use docker ```docker ps -a ps -a```