const API_BASE = 'http://localhost:5000/predict';
// const API_BASE = '/api/predict';

const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');
const form = document.getElementById('riskForm');
const resultEl = document.getElementById('result');

// Smooth Scroll Function
document.addEventListener('DOMContentLoaded', function() {
  // Mengambil semua link navigasi
  const navLinks = document.querySelectorAll('nav a, .hero-cta a, .read-more, a[href^="#"]');
  
  // Menambahkan event listener untuk setiap link
  navLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      // Mengambil target dari href
      const targetId = this.getAttribute('href');
      
      // Memeriksa apakah target adalah anchor di halaman yang sama
      if (targetId && targetId.startsWith('#') && targetId.length > 1) {
        e.preventDefault();
        
        const targetElement = document.querySelector(targetId);
        
        if (targetElement) {
          // Scroll smooth ke target
          window.scrollTo({
            top: targetElement.offsetTop - 80, // Offset untuk header
            behavior: 'smooth'
          });
        }
      }
    });
  });
  
  // Counter animation function
  const counters = document.querySelectorAll('.stat-value');
  
  counters.forEach(counter => {
    const target = parseInt(counter.getAttribute('data-target'));
    const duration = 2000; // Animation duration in milliseconds
    const step = target / (duration / 30); // Update every 30ms
    let current = 0;
    
    const updateCounter = () => {
      current += step;
      if (current < target) {
        counter.textContent = Math.floor(current);
        setTimeout(updateCounter, 30);
      } else {
        counter.textContent = target;
      }
    };
    
    updateCounter();
  });
   
  // Animasi untuk service cards
  const animateOnScroll = () => {
    const serviceCards = document.querySelectorAll('.service-card');
    
    serviceCards.forEach(card => {
      const cardTop = card.getBoundingClientRect().top;
      const triggerBottom = window.innerHeight * 0.8;
      
      if (cardTop < triggerBottom) {
        card.classList.add('animate');
      }
    });
  };
  
  // Jalankan animasi saat halaman dimuat
  animateOnScroll();
  
  // Jalankan animasi saat scroll
  window.addEventListener('scroll', animateOnScroll);
});

const next1Btn = document.getElementById('next1');
const next2Btn = document.getElementById('next2');
const back2Btn = document.getElementById('back2');
const back3Btn = document.getElementById('back3');

function flip(fromEl, toEl) {
  if (!fromEl || !toEl) return;
  fromEl.classList.add('flip-exit');
  setTimeout(() => {
    fromEl.classList.remove('active');
    fromEl.classList.remove('flip-exit');
    toEl.classList.add('active');
    toEl.classList.add('flip-enter');
    setTimeout(() => toEl.classList.remove('flip-enter'), 450);
  }, 200);
}

if (next1Btn) next1Btn.addEventListener('click', () => flip(step1, step2));
if (next2Btn) next2Btn.addEventListener('click', () => flip(step2, step3));
if (back2Btn) back2Btn.addEventListener('click', () => flip(step2, step1));
if (back3Btn) back3Btn.addEventListener('click', () => flip(step3, step2));

function validateStep(stepEl) {
  if (!stepEl) return true;
  const inputs = stepEl.querySelectorAll('input, select');
  for (const el of inputs) {
    if (!el.checkValidity()) {
      el.reportValidity();
      return false;
    }
  }
  return true;
}

if (next1Btn) next1Btn.addEventListener('click', () => {
  if (validateStep(step1)) flip(step1, step2);
});
if (next2Btn) next2Btn.addEventListener('click', () => {
  if (validateStep(step2)) flip(step2, step3);
});

function showResultPopup(contentHtml) {
  resultEl.innerHTML = `
    <div class="popup-content">
      <button class="popup-close" id="popupClose">✖</button>
      ${contentHtml}
    </div>
  `;
  resultEl.style.display = 'flex';

  const closeBtn = document.getElementById('popupClose');
  if (closeBtn) closeBtn.addEventListener('click', () => {
    resultEl.style.display = 'none';
    resetFormSteps();
  });

  const againBtn = document.getElementById('analyzeAgain');
  if (againBtn) againBtn.addEventListener('click', () => {
    resultEl.style.display = 'none';
    resetFormSteps();
  });
}

function resetFormSteps() {
  form.reset();
  step2.classList.remove('active');
  step3.classList.remove('active');
  step1.classList.add('active');
}

if (form) form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!validateStep(step3)) return;

  const data = Object.fromEntries(new FormData(form).entries());

  // convert numeric
  const numeric = ['age', 'menarche', 'agefirst', 'children', 'nrelbc', 'imc', 'weight'];
  for (const key of numeric) {
    data[key] = Number(data[key]) || 0;
  }

  // convert binary
  const binaries = ['menopause', 'breastfeeding', 'exercise', 'alcohol', 'tobacco', 'allergies'];
  for (const key of binaries) data[key] = Number(data[key]);

  try {
    const payload = { data: [
      data.age, data.menarche, data.menopause, data.agefirst,
      data.children, data.breastfeeding, data.nrelbc, data.imc,
      data.weight, data.exercise, data.alcohol, data.tobacco,
      data.allergies
    ]};

    const res = await fetch(API_BASE, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const json = await res.json();

    // Format respons dari backend
    const pred = json.pred;
    const proba = json.proba;
    const risk_category = json.risk_category;
    const level = risk_category === 'high' ? 'High' : (risk_category === 'medium' ? 'Moderate' : 'Low');
    const recommendations = json.recommendations || { lifestyle: [], medical: [] };

    // Membuat HTML untuk rekomendasi
    const lifestyleRecommendations = recommendations.lifestyle.map(rec => `<li>${rec}</li>`).join('');
    const medicalRecommendations = recommendations.medical.map(rec => `<li>${rec}</li>`).join('');

    showResultPopup(`
      <div class="result-container">
        <h3>Hasil Prediksi</h3>
        <div class="result-item"><strong>Prediksi:</strong> ${pred === 1 ? 'Berisiko' : 'Tidak Berisiko'}</div>
        <div class="result-item"><strong>Probabilitas:</strong> ${(proba * 100).toFixed(1)}%</div>
        <div class="result-item"><strong>Tingkat Risiko:</strong> <span class="risk-${risk_category}">${level}</span></div>
        
        <div class="recommendations-section">
          <h4>Rekomendasi Gaya Hidup</h4>
          <ul class="recommendations-list">
            ${lifestyleRecommendations || '<li>Tetap pertahankan gaya hidup sehat Anda</li>'}
          </ul>
          
          <h4>Rekomendasi Medis</h4>
          <ul class="recommendations-list">
            ${medicalRecommendations || '<li>Lakukan pemeriksaan rutin sesuai anjuran dokter</li>'}
          </ul>
        </div>
      </div>
      <div class="actions" style="margin-top:12px;">
        <button id="analyzeAgain" class="btn btn-outline">Analisis lagi</button>
      </div>
    `);
  } catch (err) {
    showResultPopup(`
      <div>Network error</div>
      <div class="actions"><button id="analyzeAgain" class="btn btn-outline">Analisis lagi</button></div>
    `);
  }
});
