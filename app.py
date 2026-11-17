import os
import struct
import tempfile
import numpy as np
import scipy.signal as signal
import librosa
import streamlit as st

st.set_page_config(page_title="Relevé Musical", page_icon="🎵", layout="wide")
st.title("🎵 Application de Relevé de Morceaux")
st.markdown("Importe un fichier audio pour extraire **tonalité, tempo, grille d'accords, ligne de basse** et télécharger le **MIDI**.")

NOTES_FR = ["Do (C)", "Do# (C#)", "Ré (D)", "Ré# (Eb)", "Mi (E)", "Fa (F)",
            "Fa# (F#)", "Sol (G)", "Sol# (Ab)", "La (A)", "La# (Bb)", "Si (B)"]
NOTES_EN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

def detect_key(chroma):
    maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    mi = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    mc = chroma.mean(axis=1)
    bs, bk = -1, ""
    for i in range(12):
        s1 = np.corrcoef(mc, np.roll(maj, i))[0,1]
        s2 = np.corrcoef(mc, np.roll(mi, i))[0,1]
        if s1 > bs: bs, bk = s1, f"{NOTES_FR[i]} Majeur"
        if s2 > bs: bs, bk = s2, f"{NOTES_FR[i]} mineur"
    return bk

def detect_chords(y, sr, beats):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tpl = {}
    for i, r in enumerate(NOTES_EN):
        m = np.zeros(12); m[[i,(i+4)%12,(i+7)%12]] = 1
        n = np.zeros(12); n[[i,(i+3)%12,(i+7)%12]] = 1
        tpl[r] = m/np.linalg.norm(m)
        tpl[f"{r}m"] = n/np.linalg.norm(n)
    bars = beats[::4] if len(beats) >= 4 else np.arange(0, chroma.shape[1], 86)
    bt = librosa.frames_to_time(bars, sr=sr)
    grille = []
    for idx in range(len(bars)):
        sf = bars[idx]
        ef = bars[idx+1] if idx+1 < len(bars) else chroma.shape[1]
        if ef <= sf: continue
        v = chroma[:, sf:ef].mean(axis=1)
        if v.max() < 0.05: continue
        v = v/(np.linalg.norm(v)+1e-9)
        bc = max(tpl.items(), key=lambda x: float(np.dot(v, x[1])))[0]
        t = bt[idx]
        grille.append({"M": idx+1, "T": f"{int(t//60):02d}:{int(t%60):02d}", "A": bc})
    return grille

def extract_bass(y, sr, beats):
    sos = signal.butter(4, 260, btype="low", fs=sr, output="sos")
    yb = signal.sosfilt(sos, y)
    f0 = librosa.yin(yb, fmin=librosa.note_to_hz("E1"), fmax=librosa.note_to_hz("C4"), sr=sr)
    bt = librosa.frames_to_time(beats, sr=sr)
    ev = []
    for i, bf in enumerate(beats):
        if bf < len(f0) and np.isfinite(f0[bf]) and f0[bf] > 30:
            midi = int(round(librosa.hz_to_midi(f0[bf])))
            dur = (bt[i+1]-bt[i]) if i+1 < len(bt) else 0.5
            ev.append((bt[i], dur, midi, librosa.midi_to_note(midi)))
    return ev

def make_midi(events, bpm):
    def vlq(v):
        v = max(0, int(v)); o = [v & 0x7F]; v >>= 7
        while v > 0: o.append((v & 0x7F)|0x80); v >>= 7
        return bytes(reversed(o))
    tpb = 480; spt = (60.0/max(bpm,40))/tpb; tu = int(6e7/max(bpm,40))
    tr = bytearray(b"\x00\xff\x51\x03" + tu.to_bytes(3,"big"))
    lt = 0
    for s, d, p, _ in events:
        st = int(s/spt); dt = max(60, int(d/spt)); dl = max(0, st-lt)
        p = max(21, min(108, p))
        tr += vlq(dl) + bytes([0x90, p, 90]) + vlq(dt) + bytes([0x80, p, 0])
        lt = st + dt
    tr += b"\x00\xff\x2f\x00"
    return b"MThd" + struct.pack(">IHHH",6,0,1,tpb) + b"MTrk" + struct.pack(">I",len(tr)) + bytes(tr)

uploaded = st.file_uploader("📂 Choisis un morceau (MP3, WAV, FLAC, M4A, OGG)", type=["mp3","wav","flac","m4a","ogg"])
if uploaded:
    st.audio(uploaded)
    if st.button("🚀 Faire le relevé", type="primary", use_container_width=True):
        sfx = os.path.splitext(uploaded.name)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as tmp:
            tmp.write(uploaded.getvalue()); tp = tmp.name
        try:
            with st.spinner("🎧 Analyse en cours..."):
                y, sr = librosa.load(tp, sr=22050, mono=True)
                dur = librosa.get_duration(y=y, sr=sr)
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                bpm = round(float(np.atleast_1d(tempo)[0]), 1)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                key = detect_key(chroma)
                grille = detect_chords(y, sr, beats)
                bass = extract_bass(y, sr, beats)
                midi = make_midi(bass, bpm)
            c1, c2, c3 = st.columns(3)
            c1.metric("🎼 Tonalité", key)
            c2.metric("🥁 Tempo", f"{bpm} BPM")
            c3.metric("⏱️ Durée", f"{int(dur//60)}m{int(dur%60):02d}s")
            st.subheader("🎸 Grille d'accords")
            if grille:
                cols = st.columns(4)
                for i, g in enumerate(grille):
                    cols[i%4].info(f"**Mes.{g['M']}** ({g['T']})\n### {g['A']}")
            st.subheader("🎸 Ligne de basse")
            if bass:
                st.markdown(" → ".join([f"**{b[3]}**" for b in bass[:24]]))
                st.download_button("🎹 Télécharger le MIDI", midi, f"{os.path.splitext(uploaded.name)[0]}_basse.mid", "audio/midi")
            st.success("✅ Relevé terminé !")
        finally:
            if os.path.exists(tp): os.remove(tp)
              st.divider()
st.caption("© 2026 TSHIMANGA M'PUNGA — Tous droits réservés")          
