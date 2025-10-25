

<h1>DeepFake Detector</h1>

<p>A web application to detect <strong>real vs fake images</strong> using a <strong>CNN + SVM hybrid model</strong>.</p>

<h2>Step 1: Clone or Fork the Repository</h2>
<p>Clone the repo using:</p>
<pre><code>git clone https://github.com/yourusername/ML_Project.git
cd ML_Project</code></pre>
<p>Or fork the repository on GitHub and then clone your fork.</p>

<h2>Step 2: Install Dependencies</h2>
<p>Make sure you have Python installed. Then install the required packages:</p>
<pre><code>pip install -r requirements.txt</code></pre>


<h2>Step 3: Train the Models</h2>
<p>Run the training script to train CNN and SVM models:</p>
<pre><code>python train_model.py</code></pre>
<p>This will create the <code>model/</code> folder and save:</p>
<ul>
  <li><code>cnn_model.h5</code> — Trained CNN model</li>
  <li><code>svm_model.joblib</code> — SVM trained on CNN features</li>
</ul>

<h2>Step 4: Run the Web Application</h2>
<p>Start the Flask app:</p>
<pre><code>python app.py</code></pre>
<p>Open your browser and go to:</p>
<pre><code>http://127.0.0.1:5000</code></pre>
<p>Upload an image to see:</p>
<ul>
  <li><strong>Prediction:</strong> Real or Fake</li>
  <li><strong>Confidence Score:</strong> Average of CNN and SVM predictions</li>
</ul>

</body>
</html>

