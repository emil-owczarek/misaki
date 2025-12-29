"""
Polish G2P module for misaki
Based on espeak-ng backend with Polish-specific phoneme mappings.

Phoneme inventory:
- Vowels: a, ɛ, i, ɔ, u, ɨ (+ nasal: ɔ̃, ɛ̃)
- Consonants: p, b, t, d, k, ɡ, f, v, s, z, ʂ, ʐ, ɕ, ʑ, x, m, n, ɲ, ŋ, r, l, w, j
- Affricates: ʦ, ʣ, ʧ, ʤ, ʨ, ʥ
"""

from phonemizer.backend.espeak.wrapper import EspeakWrapper
from typing import Tuple, Optional
import re

try:
    import espeakng_loader
    try:
        EspeakWrapper.set_library(espeakng_loader.get_library_path())
    except Exception:
        pass  # Library already set or not needed
    try:
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    except AttributeError:
        pass  # Older phonemizer version doesn't have set_data_path
except ImportError:
    pass  # espeak-ng must be installed system-wide

import phonemizer

# Polish-specific phoneme mappings (espeak output -> Kokoro-compatible)
PL_PHONEME_MAP = sorted({
    # Affricates with tie bars -> single symbols
    't^s': 'ʦ',
    'd^z': 'ʣ',
    't^ʃ': 'ʧ',
    'd^ʒ': 'ʤ',
    't^ɕ': 'ʨ',
    'd^ʑ': 'ʥ',
    # Alternative representations
    'tʃ': 'ʧ',
    'dʒ': 'ʤ',
    'ts': 'ʦ',
    'dz': 'ʣ',
    'tɕ': 'ʨ',
    'dʑ': 'ʥ',
    # Nasal vowels normalization
    'ɛ̃': 'ɛ̃',
    'ɔ̃': 'ɔ̃',
    # Common substitutions for better TTS
    'ɨ': 'ɨ',  # keep Polish 'y'
    'ʐ': 'ʐ',  # retroflex z (rz, ż)
    'ʂ': 'ʂ',  # retroflex s (sz)
    'ɕ': 'ɕ',  # palatal s (ś, si)
    'ʑ': 'ʑ',  # palatal z (ź, zi)
    'ɲ': 'ɲ',  # palatal n (ń, ni)
    # Remove length markers (Polish doesn't have phonemic length)
    'ː': '',
}.items(), key=lambda kv: -len(kv[0]))


class PLCleaner:
    """Text normalizer for Polish."""

    # Common abbreviations
    ABBREVIATIONS = {
        'dr': 'doktor',
        'prof.': 'profesor',
        'mgr': 'magister',
        'inż.': 'inżynier',
        'ul.': 'ulica',
        'al.': 'aleja',
        'pl.': 'plac',
        'nr': 'numer',
        'tel.': 'telefon',
        'godz.': 'godzina',
        'min.': 'minuta',
        'sek.': 'sekunda',
        'tys.': 'tysięcy',
        'mln': 'milionów',
        'mld': 'miliardów',
        'zł': 'złotych',
        'gr': 'groszy',
        'r.': 'roku',
        'w.': 'wieku',
        'n.e.': 'naszej ery',
        'p.n.e.': 'przed naszą erą',
        'itd.': 'i tak dalej',
        'itp.': 'i tym podobne',
        'np.': 'na przykład',
        'tj.': 'to jest',
        'tzn.': 'to znaczy',
        'tzw.': 'tak zwany',
        'wg': 'według',
        'ws.': 'w sprawie',
        'ds.': 'do spraw',
        'ok.': 'około',
        'ca': 'około',
        'św.': 'święty',
    }

    def __call__(self, text: str) -> str:
        """Clean and normalize Polish text."""
        text = text.strip()

        # Normalize unicode
        text = self._normalize_unicode(text)

        # Expand abbreviations
        text = self._expand_abbreviations(text)

        # Handle numbers (basic)
        # Note: num2words handles this better if installed

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Polish characters."""
        # Ensure consistent encoding for Polish diacritics
        replacements = {
            'ą': 'ą', 'ć': 'ć', 'ę': 'ę', 'ł': 'ł',
            'ń': 'ń', 'ó': 'ó', 'ś': 'ś', 'ź': 'ź', 'ż': 'ż',
            'Ą': 'Ą', 'Ć': 'Ć', 'Ę': 'Ę', 'Ł': 'Ł',
            'Ń': 'Ń', 'Ó': 'Ó', 'Ś': 'Ś', 'Ź': 'Ź', 'Ż': 'Ż',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Polish abbreviations."""
        words = text.split()
        result = []
        for word in words:
            word_lower = word.lower().rstrip('.,;:!?')
            if word_lower in self.ABBREVIATIONS:
                result.append(self.ABBREVIATIONS[word_lower])
            else:
                result.append(word)
        return ' '.join(result)


class PLG2P:
    """
    Polish Grapheme-to-Phoneme converter.

    Uses espeak-ng backend with Polish-specific phoneme mappings
    optimized for Kokoro TTS.

    Example:
        >>> g2p = PLG2P()
        >>> phonemes, _ = g2p("Dzień dobry!")
        >>> print(phonemes)  # ʥɛɲ dɔbrɨ
    """

    def __init__(self, fallback: Optional[str] = None, version: str = '2.0'):
        """
        Initialize Polish G2P.

        Args:
            fallback: Not used (espeak handles everything)
            version: espeak version for compatibility ('2.0' recommended)
        """
        self.version = version
        self.cleaner = PLCleaner()

        # Initialize espeak backend for Polish
        self.backend = phonemizer.backend.EspeakBackend(
            language='pl',
            preserve_punctuation=True,
            with_stress=True,
            tie='^',
            language_switch='remove-flags'
        )

        # Try to import num2words for number handling
        try:
            from num2words import num2words
            self.num2words = lambda n: num2words(n, lang='pl')
        except ImportError:
            self.num2words = None

    def __call__(self, text: str) -> Tuple[str, None]:
        """
        Convert Polish text to phonemes.

        Args:
            text: Polish text to convert

        Returns:
            Tuple of (phoneme_string, None)
        """
        # Clean text
        text = self.cleaner(text)

        # Convert numbers if num2words available
        if self.num2words:
            text = self._convert_numbers(text)

        # Handle quotes
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace('„', '"').replace('"', '"')

        # Phonemize
        ps = self.backend.phonemize([text])
        if not ps:
            return '', None

        ps = ps[0].strip()

        # Apply Polish-specific mappings
        for old, new in PL_PHONEME_MAP:
            ps = ps.replace(old, new)

        # Clean up
        ps = ps.replace('^', '')  # Remove tie characters

        # Handle syllabic consonants
        if self.version == '2.0':
            ps = ps.replace(chr(809), '').replace(chr(810), '')
            ps = re.sub(r'(\S)\u0329', r'ᵊ\1', ps)
        else:
            ps = ps.replace('-', '')

        return ps, None

    def _convert_numbers(self, text: str) -> str:
        """Convert numbers to Polish words."""
        def replace_number(match):
            num = match.group(0)
            try:
                # Handle integers
                if '.' not in num and ',' not in num:
                    return self.num2words(int(num))
                # Handle decimals
                else:
                    num = num.replace(',', '.')
                    return self.num2words(float(num))
            except:
                return num

        # Match numbers (including decimals with comma)
        return re.sub(r'\d+(?:[.,]\d+)?', replace_number, text)

    def phonemize(self, text: str) -> str:
        """Alias for __call__ returning just phonemes."""
        ps, _ = self(text)
        return ps


# Convenience function
def g2p(text: str) -> str:
    """Quick Polish G2P conversion."""
    return PLG2P()(text)[0]


# Test when run directly
if __name__ == '__main__':
    g2p = PLG2P()

    test_sentences = [
        "Dzień dobry!",
        "Jak się masz?",
        "Trzysta trzydzieści trzy.",
        "Książka leży na stole.",
        "Chrząszcz brzmi w trzcinie.",
        "Grzegorz Brzęczyszczykiewicz.",
        "W Szczebrzeszynie chrząszcz brzmi w trzcinie.",
    ]

    print("Polish G2P Test Results:")
    print("=" * 60)

    for sentence in test_sentences:
        phonemes, _ = g2p(sentence)
        print(f"Text: {sentence}")
        print(f"IPA:  {phonemes}")
        print()
