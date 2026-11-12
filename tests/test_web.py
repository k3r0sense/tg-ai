import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from tg_ai.web_tools import WebAgent, validate_url, PublicResolver, fetch_page


@tool
async def lookup(query: str) -> str:
    """Test lookup."""
    return 'Источник: https://example.com; найдено: ' + query


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return next(self.responses)


class WebTests(unittest.IsolatedAsyncioTestCase):
    def test_reject_private_urls(self):
        for url in ['http://127.0.0.1', 'http://[::1]', 'http://10.1.2.3',
                    'file:///etc/passwd', 'http://localhost', 'https://example.com:8080',
                    'https://user:pass@example.com']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_url(url)
        self.assertEqual(validate_url('https://example.com'), 'https://example.com')

    async def test_tool_result_reaches_model(self):
        model = FakeModel([
            AIMessage(content='', tool_calls=[{'name':'lookup', 'args':{'query':'MLX'}, 'id':'one'}]),
            AIMessage(content='Нашёл источник'),
        ])
        result = await WebAgent(model, [lookup]).ainvoke([HumanMessage(content='найди MLX')])
        self.assertEqual(result, 'Нашёл источник')
        self.assertEqual(model.calls[-1][-2].type, 'tool')
        self.assertEqual(model.calls[-1][-2].tool_call_id, 'one')
        self.assertIn('example.com', model.calls[-1][-2].content)

    async def test_tool_limit(self):
        model = FakeModel([
            AIMessage(content='', tool_calls=[{'name':'lookup', 'args':{'query':'x'}, 'id':'1'},
                                            {'name':'lookup', 'args':{'query':'y'}, 'id':'2'}]),
            AIMessage(content='Итог'),
        ])
        self.assertEqual(await WebAgent(model, [lookup], max_calls=1).ainvoke([]), 'Итог')
        self.assertIn('Лимит', model.calls[-1][-2].content)

    async def test_unknown_tool_becomes_error_result(self):
        model = FakeModel([
            AIMessage(content='', tool_calls=[{'name':'shell', 'args':{}, 'id':'1'}]),
            AIMessage(content='Не получилось'),
        ])
        await WebAgent(model, [lookup]).ainvoke([])
        self.assertIn('не выполнился', model.calls[-1][-2].content)

    async def test_dns_private_rejected(self):
        with patch('aiohttp.resolver.ThreadedResolver.resolve', new=AsyncMock(return_value=[{'host':'127.0.0.1'}])):
            with self.assertRaises(ValueError):
                await PublicResolver().resolve('example.com')

    async def test_gemma_raw_call_executes(self):
        from tg_ai.web_tools import normalize_tool_response
        response = normalize_tool_response(AIMessage(content='web_search{query:<|"|>погода Москва<|"|>}'))
        self.assertEqual(response.tool_calls[0]['args'], {'query': 'погода Москва'})
        search = AsyncMock(return_value='Найден источник')
        search.name = 'web_search'
        # Use a real tool schema with a mocked execution method.
        from tg_ai.web_tools import web_search
        model = FakeModel([
            AIMessage(content='web_search{query:<|"|>погода Москва<|"|>}'),
            AIMessage(content='Ответ по найденному источнику'),
        ])
        with patch.object(type(web_search), 'ainvoke', new=AsyncMock(return_value='Найден источник')) as invoke:
            result = await WebAgent(model, [web_search]).ainvoke([])
            invoke.assert_awaited_once_with({'query': 'погода Москва'})
        self.assertEqual(result, 'Ответ по найденному источнику')
        self.assertEqual(model.calls[-1][-2].type, 'tool')

    def test_raw_examples_are_not_executed(self):
        from tg_ai.web_tools import normalize_tool_response, final_text
        for text in ['Пример: web_search{query:<|"|>x<|"|>}',
                     'shell{query:<|"|>x<|"|>}',
                     'web_search{url:<|"|>x<|"|>}']:
            self.assertFalse(normalize_tool_response(AIMessage(content=text)).tool_calls)
        self.assertNotIn('<|', final_text(AIMessage(content='web_search{query:<|"|>broken}')))

    async def test_search_failure_is_not_reinterpreted_by_model(self):
        from tg_ai.web_tools import web_search
        model = FakeModel([AIMessage(content='', tool_calls=[
            {'name':'web_search', 'args':{'query':'кек'}, 'id':'one'}])])
        with patch.object(type(web_search), 'ainvoke', new=AsyncMock(side_effect=TimeoutError)):
            result = await WebAgent(model, [web_search]).ainvoke([])
        self.assertIn('ошибка поиска', result)
        self.assertEqual(len(model.calls), 1)

    def test_plain_json_tool_call(self):
        from tg_ai.web_tools import normalize_tool_response
        response = normalize_tool_response(AIMessage(content='web_search{"query":"значение кек"}'))
        self.assertEqual(response.tool_calls[0]['args']['query'], 'значение кек')

    async def test_region_not_sent_to_search(self):
        from tg_ai.web_tools import web_search
        with patch('tg_ai.web_tools.DDGS') as ddgs:
            ddgs.return_value.text.return_value = []
            await web_search.ainvoke({'query': 'значение кек'})
            self.assertNotIn('region', ddgs.return_value.text.call_args.kwargs)
            self.assertNotIn('region', web_search.args)

    def test_gemma_query_and_region(self):
        from tg_ai.web_tools import normalize_tool_response
        for raw in [
            'web_search{query:<|"|>популярные песни<|"|>,region:<|"|>ru-ru<|"|>}',
            '<|tool_call|>web_search{region:<|"|>ru-ru<|"|>, query:<|"|>песни<|"|>}<|/tool_call|>',
        ]:
            response = normalize_tool_response(AIMessage(content=raw))
            self.assertNotIn('region', response.tool_calls[0]['args'])
            self.assertIn('песни', response.tool_calls[0]['args']['query'])
        for raw in [
            'web_search{query:<|"|>x<|"|>,query:<|"|>y<|"|>}',
            'web_search{query:<|"|>x<|"|>,evil:123}',
        ]:
            self.assertFalse(normalize_tool_response(AIMessage(content=raw)).tool_calls)

    async def test_explicit_search_does_not_depend_on_model(self):
        from tg_ai.web_tools import web_search
        model = FakeModel([AIMessage(content='Вот источники')])
        with patch.object(type(web_search), 'ainvoke', new=AsyncMock(return_value='[]')) as invoke:
            await WebAgent(model, [web_search]).ainvoke([], forced_query='популярные треки Кендрика')
            invoke.assert_awaited_once_with({'query':'популярные треки Кендрика'})
        self.assertEqual(model.calls[0][0].tool_calls[0]['name'], 'web_search')

    def test_explicit_request_detection(self):
        from tg_ai.web_tools import explicit_search_query
        self.assertEqual(explicit_search_query('найди в сети популярные треки кендрика'), 'популярные треки кендрика')
        self.assertEqual(explicit_search_query('Можешь найти его песни?'), 'его песни?')
        self.assertIsNone(explicit_search_query('не ищи в сети'))
        self.assertIsNone(explicit_search_query('он написал: найди в сети'))

    def test_disabled_backend_cannot_fallback_silently(self):
        from tg_ai.web_tools import configured_backend
        from types import SimpleNamespace
        with patch('tg_ai.web_tools.SearchSettings', return_value=SimpleNamespace(search_backend='bing')):
            with patch.dict('tg_ai.web_tools.ENGINES', {'text': {'duckduckgo': object()}}):
                with self.assertRaises(ValueError):
                    configured_backend()

    async def test_auto_fallback(self):
        from tg_ai.web_tools import search_with_fallback
        with patch('tg_ai.web_tools._last_success', None), patch('tg_ai.web_tools._failed_until', {}):
            with patch('tg_ai.web_tools.ENGINES', {'text': {'yahoo':object(),'brave':object()}}):
                with patch('tg_ai.web_tools.DDGS') as ddgs:
                    ddgs.return_value.text.side_effect = [TimeoutError(), [{'title':'OK','href':'https://example.com'}]]
                    backend, results = await search_with_fallback('test','auto')
                    self.assertEqual(backend,'brave')
                    self.assertTrue(results)
                    self.assertEqual([c.kwargs['backend'] for c in ddgs.return_value.text.call_args_list], ['yahoo','brave'])

    async def test_cooldown_skips_failed_engine(self):
        import time
        from tg_ai.web_tools import search_with_fallback
        with patch('tg_ai.web_tools._last_success', None), patch('tg_ai.web_tools._failed_until', {'yahoo':time.monotonic()+300}):
            with patch('tg_ai.web_tools.ENGINES', {'text': {'yahoo':object(),'brave':object()}}):
                with patch('tg_ai.web_tools.DDGS') as ddgs:
                    ddgs.return_value.text.return_value = [{'title':'OK'}]
                    backend,_ = await search_with_fallback('test','auto')
                    self.assertEqual(backend,'brave')
                    ddgs.return_value.text.assert_called_once()

    def test_screenshot_request_and_followup(self):
        from tg_ai.web_tools import explicit_search_query, search_query_for_turn
        request = 'А можешь найти информацию актуальную о кендрике ламаре?'
        self.assertIsNotNone(explicit_search_query(request))
        rows = [{'text':request,'author':'Gregory','is_bot':False},
                {'text':'Да, последние','author':'Gregory','is_bot':False}]
        self.assertEqual(search_query_for_turn(rows),explicit_search_query(request))
        rows[-1]['text']='Ну давай, ищи'
        self.assertEqual(search_query_for_turn(rows),explicit_search_query(request))
        rows[-1]['author']='other'
        self.assertIsNone(search_query_for_turn(rows))

    async def test_no_tool_no_search_claim(self):
        for text in ['Поищу. Дай секунду.', 'Я искал, но инструмент не вернул мне нужных данных.',
                     'Поиск не дал конкретных результатов.']:
            model=FakeModel([AIMessage(content=text)])
            result=await WebAgent(model,[lookup]).ainvoke([])
            self.assertIn('поиск не запускался',result)
