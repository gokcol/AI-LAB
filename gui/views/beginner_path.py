"""A complete, short English/Turkish path; calculations use the same code in both languages."""
import numpy as np
import streamlit as st
from gui import i18n, progress
from urllib.parse import urlencode

tr = i18n.lang() == 'tr'
def t(en, turkish):
    return turkish if tr else en

st.title(t('Beginner path · from a neuron to evaluation', 'Başlangıç yolu · nörondan değerlendirmeye'))
st.caption(t('Four sessions, about 60 minutes. Prerequisites: addition, multiplication and basic Python expressions.',
             'Dört oturum, yaklaşık 60 dakika. Ön koşullar: toplama, çarpma ve temel Python ifadeleri.'))
if i18n.MULTILINGUAL:
    st.info(t('Every step is available in English and Turkish. Updated 2026-09-20. Human review: pending for both languages. Turkish is an AI-generated translation.',
              'Her adım İngilizce ve Türkçe sunulur. Güncelleme: 2026-09-20. İki dil için de insan incelemesi bekleniyor. Türkçe, yapay zekâ çevirisidir.'))
else:
    st.info("Human review: pending.")
names = [t('1 · One neuron', '1 · Tek nöron'), t('2 · Learn one weight', '2 · Bir ağırlığı öğren'),
         t('3 · Build XOR', '3 · XOR kur'), t('4 · Evaluate honestly', '4 · Dürüstçe değerlendir')]
step = st.radio(t('Session', 'Oturum'), range(4), format_func=lambda i: names[i], horizontal=True)
topic = ['playground', 'backprop', 'two_neurons', 'ml_classification'][step]
if step == 0:
    st.write(t('Begin with the 20-minute neuron lesson. Predict the output, move the bias, calculate one answer, then explain it in your own words. Return here for session 2.',
               'İki dilde sunulan 20 dakikalık nöron dersiyle başla. Çıktıyı tahmin et, bias değerini değiştir, bir sonuç hesapla ve kendi sözlerinle açıkla. Ardından 2. oturum için buraya dön.'))
    st.page_link('views/start_here.py', label=t('Open session 1', '1. oturumu aç'))
elif step == 1:
    st.write(t('A neuron predicts ŷ = w·x. Take x=2, target y=6 and initial w=1. The loss is L=(ŷ−y)²/2. Learning changes w to reduce this loss.',
               'Nöronun tahmini ŷ = w·x olsun. x=2, hedef y=6 ve başlangıç ağırlığı w=1 seçilsin. Kayıp L=(ŷ−y)²/2 olur. Öğrenme, bu kaybı azaltmak için w değerini değiştirir.'))
    st.latex(r'\frac{\partial L}{\partial w}=(wx-y)x,\qquad w_{new}=w-\eta\frac{\partial L}{\partial w}')
    value = st.number_input(t('Calculate the gradient at w=1', 'w=1 için gradyanı hesapla'), value=None, step=1.)
    level = st.session_state.get('begin_gradient_hint', 0)
    if st.button(t('Next hint', 'Sonraki ipucu'), disabled=level >= 2):
        level += 1
        st.session_state.begin_gradient_hint = level
    if level >= 1:
        st.info(t('First calculate prediction minus target: 1·2−6.', 'Önce tahmin eksi hedefi hesapla: 1·2−6.'))
    if level >= 2:
        st.info(t('Multiply that error by the input x=2. Keep the negative sign.', 'Bu hatayı x=2 girdisiyle çarp. Negatif işareti koru.'))
    if value is not None:
        (st.success if abs(value+8)<1e-8 else st.warning)(t('Correct: −8.' if abs(value+8)<1e-8 else 'Try again using the chain rule.',
                                                        'Doğru: −8.' if abs(value+8)<1e-8 else 'Zincir kuralını kullanarak tekrar dene.'))
    st.write(t('Implementation challenge: complete the SGD update. The gradient is supplied; choose its multiplier.',
               'Uygulama alıştırması: SGD güncellemesini tamamla. Gradyan verilmiş durumda; çarpanını seç.'))
    st.code('new_w = w + coefficient * learning_rate * gradient', language='python')
    coefficient = st.number_input(t('coefficient', 'katsayı'), value=None, step=1.)
    if st.button(t('Check on two gradients', 'İki gradyanla kontrol et'), disabled=coefficient is None):
        passed = np.allclose(1 + coefficient * .1 * np.array([-8., 2.]), [1.8, .8])
        (st.success if passed else st.warning)(t('Both update checks pass.' if passed else 'The update must move opposite to the gradient.',
                                               'İki güncelleme kontrolü de geçti.' if passed else 'Güncelleme gradyanın ters yönünde olmalı.'))
    st.caption(t('Scope: one sample, one weight, squared loss; no bias or stochastic minibatches.',
                 'Kapsam: tek örnek, tek ağırlık ve karesel kayıp; bias veya rastgele mini grup yok.'))
elif step == 2:
    st.write(t('XOR outputs 1 only when exactly one input is 1. A single straight boundary cannot separate its four points. Build two hidden features: OR and AND.',
               'XOR, yalnızca girdilerden tam biri 1 olduğunda 1 üretir. Tek bir doğru, dört noktayı ayıramaz. İki gizli özellik kur: OR ve AND.'))
    st.code('h_or  = int(x1 + x2 >= 0.5)\nh_and = int(x1 + x2 >= 1.5)\noutput = int(h_or + coefficient * h_and >= 0.5)', language='python')
    coefficient = st.number_input(t('Choose the coefficient for h_and', 'h_and için katsayıyı seç'), value=None, step=.5)
    if st.button(t('Test all four inputs', 'Dört girdiyi de dene'), disabled=coefficient is None):
        x = np.array([[0,0], [0,1], [1,0], [1,1]])
        h_or = (x.sum(1)>=.5).astype(int)
        h_and = (x.sum(1)>=1.5).astype(int)
        predicted = (h_or+coefficient*h_and>=.5).astype(int)
        st.dataframe({'x1':x[:,0], 'x2':x[:,1], t('Output','Çıktı'):predicted,
                      t('Target','Hedef'):[0,1,1,0]}, hide_index=True)
        passed = np.array_equal(predicted, [0,1,1,0])
        (st.success if passed else st.info)(t('All four pass. Explain why many negative coefficients work.',
                                               'Dördü de geçti. Birden fazla negatif katsayının neden işe yaradığını açıkla.') if passed else
                                            t('The (1,1) case activates both features. Its score must fall below 0.5.',
                                              '(1,1) girdisi iki özelliği de etkinleştirir. Puanı 0.5 değerinin altında kalmalı.'))
    with st.expander(t('Hint: inspect the hidden features', 'İpucu: gizli özellikleri incele')):
        st.write(t('OR accepts either input. Subtract enough when AND also fires: coefficient < −0.5.',
                   'OR, girdilerden birinin 1 olmasını kabul eder. AND de etkinse yeterince çıkar: katsayı < −0.5.'))
    st.caption(t('Scope: manually constructed threshold gates; this is not a trained network or a generalization test.',
                 'Kapsam: elle kurulmuş eşik kapıları; eğitilmiş ağ veya genelleme testi değil.'))
else:
    st.write(t('Of 100 machines, 95 are healthy and 5 are faulty. A model always predicts healthy. Accuracy is 95%, but how many faults does it find?',
               '100 makinenin 95 tanesi sağlam, 5 tanesi arızalı. Model her zaman sağlam tahmini yapıyor. Doğruluk %95; peki kaç arızayı buluyor?'))
    found = st.number_input(t('Faults found', 'Bulunan arıza sayısı'), 0, 5, value=None)
    if found is not None:
        (st.success if found == 0 else st.info)(t('Correct: zero. Recall = 0/5 = 0. Accuracy hides the missed faults.' if found == 0 else 'Check which label the model ever predicts.',
                                                'Doğru: sıfır. Duyarlılık = 0/5 = 0. Doğruluk, kaçırılan arızaları gizliyor.' if found == 0 else 'Modelin hangi etiketi tahmin ettiğini kontrol et.'))
    st.write(t('To improve the model, tune its settings on validation data. Keep test data for a final check. Split repeated entities together so a memorized identity cannot leak across sets.',
               'Modeli iyileştirmek için ayarları doğrulama verisinde seç. Test verisini son kontrol için sakla. Aynı varlığa ait kayıtları birlikte ayır; ezberlenen kimlik veri kümeleri arasında sızıntı yaratmasın.'))
    evidence = st.radio(t('A document says “open 09:00 on weekdays.” What does it prove about Sunday?',
                          'Belgede “hafta içi 09:00’da açılır” yazıyor. Bu, pazar günü için neyi kanıtlar?'),
                         [t('Opens at 09:00', '09:00’da açılır'), t('Insufficient evidence', 'Yetersiz kanıt')], index=None)
    if evidence is not None:
        correct = evidence == t('Insufficient evidence', 'Yetersiz kanıt')
        (st.success if correct else st.info)(t('Correct: weekday evidence does not establish Sunday hours.' if correct else 'Does the evidence mention Sunday?',
                                               'Doğru: hafta içi bilgisi pazar saatlerini kanıtlamaz.' if correct else 'Kanıtta pazar günü geçiyor mu?'))
    st.caption(t('Scope: synthetic counts and one document; this does not measure the reliability of a deployed system.',
                 'Kapsam: yapay sayımlar ve tek belge; çalışan bir sistemin güvenilirliğini ölçmez.'))

if step > 0:
    with st.form(f'beginner_checkpoint_{step}'):
        note = st.text_area(t('Explain without looking: what did you learn, and what can break it?',
                              'Bakmadan açıkla: ne öğrendin ve hangi durumda geçersiz olur?'), max_chars=4000)
        if st.form_submit_button(t('Save my explanation', 'Açıklamamı kaydet')):
            progress.update_topic(topic, stage='Explained', bookmark=True, note=note)
            st.success(t('Saved in this browser. Export a backup from My learning.',
                         'Bu tarayıcıya kaydedildi. Öğrenimim sayfasından yedek dışa aktarabilirsin.'))
    st.page_link('views/my_learning.py', label=t('My learning and notebook (partially translated)', 'Öğrenimim ve deney defteri (kısmen çevrildi)'))
    if step < 3:
        st.caption(t('Continue with the next session above.', 'Yukarıdan sonraki oturuma geç.'))
    else:
        st.success(t('You have connected prediction, learning, hidden features and evaluation. Next, compare the same MNIST model in NumPy and PyTorch.',
                     'Tahmin, öğrenme, gizli özellikler ve değerlendirmeyi birleştirdin. Sırada aynı MNIST modelini NumPy ve PyTorch ile karşılaştırmak var.'))
        st.page_link('views/bridge_lab.py', label=t('Continue to model bridges (English)', 'Model köprülerine devam et (İngilizce)'))
if i18n.MULTILINGUAL:
    st.caption(t('Sources: the lab’s neuron, backpropagation, XOR and classification lessons. Shared equations and checks are identical in both languages.',
                 'Kaynaklar: laboratuvarın nöron, geri yayılım, XOR ve sınıflandırma dersleri. Denklemler ve kontroller iki dilde aynıdır.'))
else:
    st.caption("Sources: the lab’s neuron, backpropagation, XOR and classification lessons.")
query = urlencode({'title':'Beginner path correction', 'body':f'Lesson: beginner_path, session {step+1}\nLanguage: {"tr" if tr else "en"}\nCorrection and supporting source:\n'})
st.link_button(t('Report an issue with this lesson', 'Bu dersle ilgili sorun bildir'), 'https://github.com/gokcol/AI-LAB/issues/new?' + query)
