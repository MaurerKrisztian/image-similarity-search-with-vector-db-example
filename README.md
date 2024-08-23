# Image Similarity Search With Vector DB

### Setup Instructions

0. Use a venv (optional)
```bash
venv python -m venv venv
source venv/bin/activate
```

1. **Install the required packages**  
   - Run the following command to install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   
2. **Add Images**  
   - Place the images you want to vectorize in the `data/images` folder. These images will be processed and saved to the database.

3. **Create Embeddings**  
   - Run the script to convert images into searchable vectors:

    ```bash
   python create_embeddings.py
    ```

4. **Start the API**
   - Launch the API by running:
    ```bash
    python api.py
    ```
The API will be available at http://0.0.0.0:8000.

5. View the Application
   - Open the index.html file in your browser to access the interface.